#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pip install -r $SCRIPT_DIR/test-requirements.txt

if [ `ps -ef|grep visdom |wc -l` = "1" ]
    then
    echo `ps|grep visdom |wc -l`
    echo "no visdom"
    visdom &
    # kill visdom when done testing
    trap "kill -SIGTERM $!" EXIT
fi

pytest
