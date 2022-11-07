from kafka import KafkaProducer

import socketio

import json


sio = socketio.Client()
producer = KafkaProducer(
    bootstrap_servers=["kafka:9092"],
    value_serializer=lambda x: json.dumps(x).encode("ascii"),
    api_version=(0, 10, 2)
)


@sio.event
def connect():
    print("I'm connected!")
    sio.emit("chat", {"x": 12, "y": 42, "car_id": 2, "sin": 0.2, "is_car": True})


@sio.event
def connect_error():
    print("The connection failed!")
    sio.connect("http://soskov.online:5000")


@sio.on("chat")
def on_message(data):
    angle = data["rotate"]
    speed = data["speed"]

    print(angle)

    producer.send(
        "motors",
        value={"speed": speed, "rotation_angle": angle}
    )


@sio.event
def disconnect():
    print("disconnect")
    try:
        sio.disconnect()
        sio.connect("http://soskov.online:5000")
    except:
        print("zxc")
        sio.connect("http://soskov.online:5000")


sio.connect("http://soskov.online:5000", wait_timeout=10)
