import tensorrt as trt


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
        return builder.build_serialized_network(network, config)


if __name__ == "__main__":
    TRT_LOGGER = trt.Logger()
    engine = build_engine("../models/onnx/model.onnx", TRT_LOGGER, (3, 224, 224))
    with open("../car_workflow/tensorrt_engine.engine", "wb") as f:
        f.write(engine)
