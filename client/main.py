from kafka import KafkaProducer

import socketio
import json


sio = socketio.Client()


@sio.on("chat")
def on_message(data):
    angle = data["rotate"]
    speed = data["speed"]

    producer.send(
        "motors",
        value={"speed": int(speed), "rotation_angle": int(angle)}
    )


if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=["kafka:9092"],
        value_serializer=lambda x: json.dumps(x).encode("ascii"),
        api_version=(0, 10, 2)
    )
    sio.connect("http://192.168.100.9:3000", wait_timeout=10)
    sio.wait()
