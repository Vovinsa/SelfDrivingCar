# Base l4t container
FROM nvcr.io/nvidia/l4t-base:r34.1.1

WORKDIR /usr/src/car/utils

ADD . /usr/src/car/utils

RUN apt-get update

RUN apt-get install -y python3-opencv