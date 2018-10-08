#!/bin/bash

# Exit on error
set -e

# Delete all existing Python protobuf (*_pb2.py) output
rm -rf tensorboardX/proto/*pb2*.py


# Download protoc. Make sure we are using same version of protoc
protoc="protoc-3.6.1-linux-x86_64.zip"
if [ ! -f $protoc ]; then
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/$protoc
fi
rm -rf protoc
python -c "import zipfile; zipfile.ZipFile('protoc-3.6.1-linux-x86_64.zip','r').extractall('protoc')"
chmod +x protoc/bin/protoc

# Regenerate
protoc/bin/protoc tensorboardX/proto/*.proto --python_out=.

echo "Done generating tensorboardX/proto/*pb2*.py"
