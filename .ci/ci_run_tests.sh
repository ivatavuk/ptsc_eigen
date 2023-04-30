#!/bin/bash
set -e

echo "Starting tests"

#Test ptsc_eigen
cd $GITHUB_WORKSPACE
./test_installation.sh

echo "Ending tests"