from kafka import KafkaProducer

from camera import Camera

import pickle


if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=["localhost:29092"],
        value_serializer=lambda x: pickle.dumps(x),
        api_version=(0, 10, 2)
    )
    cam = Camera(capture_width=1640, capture_height=1232,
                 display_width=224, display_height=224, framerate=30)

    while True:
        frame = cam.update()
        producer.send(
            "camera",
            {"frame": frame}
        )
