from kafka import KafkaConsumer

from dcmotor import DCMotor

import json


if __name__ == "__main__":
    dc_motor = DCMotor()
    consumer = KafkaConsumer(
        "motors",
        bootstrap_servers=["kafka:9092"],
        value_deserializer=lambda x: json.loads(x.decode("ascii")),
        api_version=(0, 10, 2)
    )

    for msg in consumer:
        speed = msg.value["speed"]
        if speed == 1:
            dc_motor.forward(100)
        elif speed == -1:
            dc_motor.backward(100)
        else:
            dc_motor.forward(0)
