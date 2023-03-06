from kafka import KafkaConsumer

from servo import ServoMotor

import json


if __name__ == "__main__":
    servo_motor = ServoMotor(channel=0)
    consumer = KafkaConsumer(
        "motors",
        bootstrap_servers=["kafka:9092"],
        value_deserializer=lambda x: json.loads(x.decode("ascii")),
        api_version=(0, 10, 2)
    )

    for msg in consumer:
        angle = msg.value["rotation_angle"]
        servo_motor.set_rotation_angle(angle=angle)
