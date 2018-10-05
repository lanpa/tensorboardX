#!/bin/bash

# Exit on error
set -e

# Delete all existing Python protobuf (*_pb2.py) output
rm -rf tensorboardX/proto/*pb2*.py

# Regenerate
protoc tensorboardX/proto/*.proto --python_out=.

echo "Done generating tensorboardX/proto/*pb2*.py"
