# Base l4t container with Jetpack 4.6.1
FROM nvcr.io/nvidia/l4t-base:r34.1

WORKDIR /usr/src/car/utils/servo

ADD . /usr/src/car/utils/servo

RUN apt-get update && \
    apt-get install -y python3-pip

# Setting up Servo and DC Motor control
RUN pip3 install --upgrade pip \
    --upgrade setuptools \
    --upgrade future \
    adafruit-circuitpython-servokit==1.3.8 \
    Jetson.GPIO \
    kafka-python==2.0.2

CMD ["python3", "run.py"]