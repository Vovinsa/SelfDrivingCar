# Base l4t container with Jetpack 5.0.2
FROM nvcr.io/nvidia/l4t-base:r34.1

WORKDIR /usr/src/car/utils/dcmotor/

ADD . /usr/src/car/utils/dcmotor/

RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip3 install --upgrade pip \
    kafka-python==2.0.2 \
    Jetson.GPIO

CMD ["python3", "run.py"]