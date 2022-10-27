# Base l4t container with Jetpack 5.0.2
FROM nvcr.io/nvidia/l4t-base:r35.1.0

WORKDIR /usr/src/car/utils

ADD .. /usr/src/car/utils

RUN apt-get update

RUN apt-get install -y python3-pip && pip3 install --upgrade pip

# Setting up Servo and DC Motor control
RUN pip3 install Jetson.GPIO