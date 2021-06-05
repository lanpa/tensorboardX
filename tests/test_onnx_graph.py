import unittest
import torch
from tensorboardX import SummaryWriter

# https://github.com/onnx/models/blob/master/vision/classification/mnist/model/mnist-8.onnx
class ONNXGraphTest(unittest.TestCase):
    def test_onnx_graph(self):
        with SummaryWriter() as w:
            w.add_onnx_graph('tests/mnist-8.onnx')
