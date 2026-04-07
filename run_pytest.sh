#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

uv pip install -r $SCRIPT_DIR/test-requirements.txt

pytest
