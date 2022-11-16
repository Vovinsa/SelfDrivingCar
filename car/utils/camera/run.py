from kafka import KafkaProducer

from camera import Camera


if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=["kafka:9092"]
    )
    cam = Camera(capture_width=1280, capture_height=720,
                 display_width=224, display_height=224, framerate=30)

    while True:
        frame = cam.update()
        producer.send(
            "camera",
            {"frame": frame}
        )
