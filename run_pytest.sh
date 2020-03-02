pip install pytest boto3 moto onnx tensorboard matplotlib 

if [ `ps|grep visdom |wc -l` = "1" ]
    then
    echo `ps|grep visdom |wc -l`
    echo "no visdom"
    visdom &
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python pytest

pytest tests/tset_multiprocess_write.py
