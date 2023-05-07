#!/bin/bash

set -e

echo "Starting build"

#Build and install ptsc_eigen
cd $GITHUB_WORKSPACE
mkdir build
cd build
cmake ..
make
sudo make install

echo "Ending build"