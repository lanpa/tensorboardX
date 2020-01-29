import unittest
from tensorboardX import SummaryWriter

class OPENVINOGraphTest(unittest.TestCase):
    def test_openvino_graph(self):
        with SummaryWriter() as w:
            w.add_openvino_graph('examples/mobilenetv2.xml')
