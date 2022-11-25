# Base l4t container with Jetpack 4.6.1
FROM nvcr.io/nvidia/l4t-base:r32.7.1

WORKDIR /usr/src/client

ADD . /usr/src/client

RUN apt-get update

RUN apt-get install -y python3-pip && pip3 install --upgrade pip

RUN pip3 install python-socketio==5.7.2 \
    websocket-client==1.3.1  \
    requests==2.18.4 \
    kafka-python==2.0.2

CMD ["python3", "main.py"]