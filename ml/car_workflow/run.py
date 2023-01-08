import numpy as np

# import onnxruntime as ort

from kafka import KafkaProducer, KafkaConsumer

from utils.infer import TensorrtModel

import pickle
import json
import time
import argparse


parser = argparse.ArgumentParser(description="Run parser")
parser.add_argument("--engine_path", type=str,
                    help="Path to the engine", default="tensorrt_engine.engine")


if __name__ == "__main__":
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=["localhost:29092"],
        value_serializer=lambda x: json.dumps(x).encode("ascii"),
        api_version=(0, 10, 2)
    )
    consumer = KafkaConsumer(
        "camera",
        bootstrap_servers=["localhost:29092"],
        value_deserializer=lambda x: pickle.loads(x),
        api_version=(0, 10, 2)
    )

    trt_model = TensorrtModel(args.engine_path, "../models/onnx/model.onnx")

    for msg in consumer:
        s = time.time()
        frame = msg.value["frame"]
        out = trt_model(frame)
        angle, speed = out[0].round().item(), out[1].round().item()
        producer.send(
            "motors",
            value={"rotation_angle": angle, "speed": speed}
        )
        print(time.time() - s)
