import json

from kafka import KafkaConsumer

from servo import ServoMotor

servo_motor = ServoMotor(channel=0)
consumer = KafkaConsumer(
    "motors",
    bootstrap_servers=["localhost:29092"],
    value_deserializer=lambda x: json.loads(x.decode("ascii")),
    api_version=(0, 10, 2)
)

for msg in consumer:
    angle = msg.value["rotation_angle"]
    servo_motor.set_rotation_angle(angle=angle)
