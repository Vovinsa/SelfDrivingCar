# Base l4t container with Jetpack 5.0.2
FROM nvcr.io/nvidia/l4t-base:r35.1.0

WORKDIR /usr/src/client

ADD . /usr/src/client

RUN apt-get update

RUN apt-get install -y python3-pip && pip3 install --upgrade pip

RUN pip3 install python-socketio==5.7.1 websocket-client==1.3.1
