#!/usr/bin/env bash
printf "Starting install..."

sudo apt-get -y install libopencv-dev build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy python-matplotlib libdc1394-22 libdc1394-22-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip libgtk2.0-dev libatlas-base-dev gfortran

git clone https://github.com/nextBillyonair/compVision.git

mkdir opencv
cd opencv
wget https://github.com/opencv/opencv/archive/3.1.0.zip
unzip 3.1.0.zip

cd opencv-3.1.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..

make -j4

sudo make install
sudo ldconfig

printf "Installation Complete..."
printf "Please Verify that OpenCV 3.1.0 has been imstalled..."
printf "Please run python -> import cv2 -> cv2.__version__ -> '3.1.0'"
