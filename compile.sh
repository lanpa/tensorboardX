#!/bin/bash

# Exit on error
set -e

# Delete all existing Python protobuf (*_pb2.py) output
rm -rf tensorboardX/proto/*pb2*.py


PROTOC_BIN=`which protoc`
if [ -z ${PROTOC_BIN} ]; then
  # Download and use the latest version of protoc.
  if [ "$(uname)" == "Darwin" ]; then
    PROTOC_ZIP="protoc-3.6.1-osx-x86_64.zip"
  else
    PROTOC_ZIP="protoc-3.6.1-linux-x86_64.zip"
  fi
  WGET_BIN=`which wget`
  if [[ ! -z ${WGET_BIN} ]]; then
    ${WGET_BIN} https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/${PROTOC_ZIP}
    rm -rf protoc
    python -c "import zipfile; zipfile.ZipFile('"${PROTOC_ZIP}"','r').extractall('protoc')"
    PROTOC_BIN=protoc/bin/protoc
    chmod +x ${PROTOC_BIN}
  fi
fi

# Regenerate
if [[ ! -z ${PROTOC_BIN} ]]; then
  ${PROTOC_BIN} tensorboardX/proto/*.proto --python_out=.

  echo "Done generating tensorboardX/proto/*pb2*.py"
else
  echo "protoc not installed so can't regenerate tensorboardX/proto/*pb2*.py"
fi

