from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
import tensorboardX.beholder as beholder_lib
import tensorboardX.beholder.file_system_tools as fio
import tempfile
from collections import namedtuple
import shutil


class BeholderTest(unittest.TestCase):
    def setUp(self):
        self.LOG_DIRECTORY = tempfile.mkdtemp()

    def tearDown(self):
        print("removing", self.LOG_DIRECTORY)
        shutil.rmtree(self.LOG_DIRECTORY)

    def test_beholder(self):
        tensor_and_name = namedtuple('tensor_and_name', 'tensor, name')
        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i)) for i in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i)) for i in range(5)]
        beholder = beholder_lib.Beholder(logdir=self.LOG_DIRECTORY)
        beholder.update(
            trainable=fake_param,
            arrays=arrays,
            frame=np.random.randn(128, 128),
        )

    def test_beholder_video(self):
        LOG_DIRECTORY = self.LOG_DIRECTORY
        tensor_and_name = namedtuple('tensor_and_name', 'tensor, name')
        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i)) for i in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i)) for i in range(5)]
        beholder = beholder_lib.Beholder(logdir=self.LOG_DIRECTORY)
        pkl = fio.read_pickle(LOG_DIRECTORY + '/plugins/beholder/config.pkl')
        pkl['is_recording'] = True
        fio.write_pickle(pkl, LOG_DIRECTORY + '/plugins/beholder/config.pkl')
        for i in range(3):
            if i == 2:
                pkl = fio.read_pickle(LOG_DIRECTORY + '/plugins/beholder/config.pkl')
                pkl['is_recording'] = False
                fio.write_pickle(pkl, LOG_DIRECTORY + '/plugins/beholder/config.pkl')
            beholder.update(
                trainable=fake_param,
                arrays=arrays,
                frame=np.random.randn(128, 128),
            )
