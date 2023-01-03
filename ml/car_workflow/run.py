import numpy as np

import onnxruntime as ort

from kafka import KafkaProducer, KafkaConsumer

import pickle
import json


import time


if __name__ == "__main__":
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

    ort_sess = ort.InferenceSession("../models/onnx/model.onnx", providers=["TensorrtExecutionProvider"])
    print("A")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for msg in consumer:
        s = time.time()
        frame = msg.value["frame"]
        frame = frame / 255
        frame = np.transpose(frame, (2, 0, 1))
        frame = (frame - mean[:, None, None]) / std[:, None, None]
        frame = np.expand_dims(frame, 0)

        ort_inp = {
            "input.img": frame.astype(np.float32),
        }
        ort_out = ort_sess.run(["output.angle", "output.speed"], ort_inp)

        angle, speed = ort_out[0].round().item(), ort_out[1].round().item()
        print(angle, speed)

        producer.send(
            "motors",
            value={"rotation_angle": angle, "speed": speed}
        )
        print(time.time() - s)
