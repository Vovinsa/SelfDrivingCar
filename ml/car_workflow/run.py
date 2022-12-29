import torch

from kafka import KafkaProducer, KafkaConsumer

import pickle
import json


if __name__ == "__main__":
    producer = KafkaProducer()
    consumer = KafkaConsumer()

    for msg in consumer:
        img = msg["frame"]
