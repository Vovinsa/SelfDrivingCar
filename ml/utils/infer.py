import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np

import os


class TensorrtModel:
    def __init__(self, engine_path, onnx_model_path):
        self.engine_path = engine_path
        self.onnx_model_path = onnx_model_path
        self.trt_logger = trt.Logger()
        if not os.path.exists(self.engine_path):
            engine = self.build_engine(self.onnx_model_path, self.trt_logger, (3, 224, 224))
            with open(self.engine_path, "wb") as f:
                f.write(engine)
        self.engine = self.load_engine(self.engine_path, self.trt_logger)
        self.context = self.engine.create_execution_context()
        self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings, self.stream = self.allocate_buffers()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def build_engine(onnx_model_path, trt_logger, img_shape, batch_size=1, silent=False):
        with trt.Builder(trt_logger) as builder, \
                builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, trt_logger) as parser:
            builder.max_batch_size = batch_size
            with open(onnx_model_path, "rb") as model:
                model_trt = parser.parse(model.read())
                if not model_trt:
                    print("ERROR: Failed to parse the ONNX file")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print("ONNX file parsed")
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.set_flag(trt.BuilderFlag.FP16)

            input = network.get_input(0)
            profile.set_shape(input.name, (batch_size, img_shape[0], img_shape[1], img_shape[2]),
                              (batch_size, img_shape[0], img_shape[1], img_shape[2]),
                              (batch_size, img_shape[0], img_shape[1], img_shape[2]))
            config.add_optimization_profile(profile)

            if not silent:
                print("Building TensorRT engine. This may take few minutes.")
            engine = builder.build_serialized_network(network, config)
            return engine

    @staticmethod
    def load_engine(engine_path, trt_logger):
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self):
        host_inputs = []
        cuda_inputs = []

        host_outputs = []
        cuda_outputs = []

        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, stream

    def __call__(self, frame):
        frame = frame.astype(np.float32)
        frame = frame / 255
        frame = np.transpose(frame, (2, 0, 1))
        frame = (frame - self.mean[:, None, None]) / self.std[:, None, None]
        np.copyto(self.host_inputs[0], frame.ravel())

        for host_inp, cuda_inp in zip(self.host_inputs, self.cuda_inputs):
            cuda.memcpy_htod_async(cuda_inp, host_inp, self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        for host_out, cuda_out in zip(self.host_outputs, self.cuda_outputs):
            cuda.memcpy_dtoh_async(host_out, cuda_out, self.stream)
        self.stream.synchronize()
        return self.host_outputs

