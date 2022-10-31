# Base l4t container with Jetpack 4.6.1
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.5.0

WORKDIR /root

RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    cmake \
    gfortran \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavresample-dev \
    libcanberra-gtk3-module \
    libdc1394-22-dev \
    libeigen3-dev \
    libglew-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev \
    dbus \
    libgtk-3-dev \
    libjpeg-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtesseract-dev \
    libtiff-dev \
    libv4l-dev \
    libxine2-dev \
    libxvidcore-dev \
    libx264-dev \
    pkg-config \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-matplotlib \
    qv4l2 \
    v4l-utils \
    v4l2ucp \
    zlib1g-dev \
    unzip

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip

RUN unzip opencv.zip && unzip opencv_contrib.zip && \
    rm -rf opencv.zip && rm -rf opencv_contrib.zip

RUN mkdir -p opencv-${OPENCV_VERSION}/build && cd opencv-${OPENCV_VERSION}/build && \
    cmake \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules ../../opencv-${OPENCV_VERSION} \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local/ \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_OPENGL=ON .. && \
    make -j1 && make install

RUN pip3 install kafka-python

WORKDIR /usr/src/car/utils/camera

ADD . /usr/src/car/utils/camera

CMD ["python3", "run.py"]