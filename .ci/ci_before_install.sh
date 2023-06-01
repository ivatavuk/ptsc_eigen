#!/bin/bash
set -e

echo "Starting install"

# get the current commit SHA
SHA=`git rev-parse HEAD`

# get the current package name
PACKAGE_NAME=${PWD##*/}

sudo apt-get -y install git

cd
git clone --recursive --branch release-0.6.3 https://github.com/osqp/osqp
cd osqp
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install

#Install Eigen3
sudo apt install libeigen3-dev

#Build and install osqp-eigen
cd
git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build
cd build
cmake ..
make
sudo make install

echo "Ending install"