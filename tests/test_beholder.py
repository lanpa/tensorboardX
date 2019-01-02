from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboardX import SummaryWriter
import numpy as np
import pytest
import unittest
import tensorboardX.beholder as beholder_lib
from collections import namedtuple


class BeholderTest(unittest.TestCase):
    def test_beholder(self):
        LOG_DIRECTORY = '/tmp/beholder-demo'
        tensor_and_name = namedtuple('tensor_and_name', 'tensor, name')


        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                        for i in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                    for i in range(5)]
        beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        beholder.update(
            trainable=fake_param,
            arrays=arrays,
            frame=np.random.randn(128, 128),
        )