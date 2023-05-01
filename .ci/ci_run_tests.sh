#!/bin/bash
set -e

echo "Starting tests"

#Test ptsc_eigen
cd $GITHUB_WORKSPACE/build
ctest

echo "Ending tests"