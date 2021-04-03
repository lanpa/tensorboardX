pip install pytest boto3 moto onnx tensorboard matplotlib flake8==3.8.3

if [ `ps|grep visdom |wc -l` = "1" ]
    then
    echo `ps|grep visdom |wc -l`
    echo "no visdom"
    visdom &
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python pytest

