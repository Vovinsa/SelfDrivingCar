import numpy as np

import onnxruntime as ort

from kafka import KafkaProducer, KafkaConsumer

import pickle
import json


if __name__ == "__main__":
    producer = KafkaProducer(
        boostrap_servers=["kafka:9092"],
        value_serializer=lambda x: json.dumps(x).encode("ascii"),
        api_version=(0, 10, 2)
    )
    consumer = KafkaConsumer(
        "camera",
        bootsrap_servers=["kafka:9092"],
        value_deserializer=lambda x: pickle.loads(x.decode("ascii")),
        api_version=(0, 10, 2)
    )

    ort_sess = ort.InferenceSession("../models/onnx/model.onnx", providers=["CUDAExecutionProvider"])
    ort_inp = {
        "input.img": np.random.rand(1, 3, 224, 224).astype(np.float32),
        "input.meas": np.array([[25, 0]], dtype=np.float32)
    }
    for msg in consumer:
        frame = msg["frame"]
        ort_out = ort_sess.run(["output.angle", "output.speed"], ort_inp)
        angle, speed = ort_out[0], ort_out[1]
        producer.send(
            "motors",
            value={"rotation_angle": angle, "speed": speed}
        )
