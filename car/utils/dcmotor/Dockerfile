# Base l4t container with Jetpack 5.0.2
FROM nvcr.io/nvidia/l4t-base:r32.7.1

WORKDIR /usr/src/car/utils

ADD .. /usr/src/car/utils

RUN apt-get update && \
    apt-get install -y python3 pip

RUN pip3 install --upgrade pip

# Setting up Servo and DC Motor control
RUN pip3 install Jetson.GPIO