from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorboardX import x2num, SummaryWriter
import torch
import numpy as np
import unittest


class PyTorchNumpyTest(unittest.TestCase):
    def test_pytorch_np(self):
        tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
        for tensor in tensors:
            # regular tensor
            assert isinstance(x2num.make_np(tensor), np.ndarray)

            # CUDA tensor
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(tensor.cuda()), np.ndarray)

            # regular variable
            assert isinstance(x2num.make_np(torch.autograd.Variable(tensor)), np.ndarray)

            # CUDA variable
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

        # python primitive type
        assert(isinstance(x2num.make_np(0), np.ndarray))
        assert(isinstance(x2num.make_np(0.1), np.ndarray))

    def test_pytorch_img(self):
        shapes = [(77, 3, 13, 7), (77, 1, 13, 7), (3, 13, 7), (1, 13, 7), (13, 7)]
        for s in shapes:
            x = torch.Tensor(np.random.random_sample(s))
            # assert x2num.make_np(x, 'IMG').shape[2] == 3

    def test_pytorch_vid(self):
        shapes = [(16, 3, 30, 28, 28), (19, 3, 30, 28, 28), (19, 3, 29, 23, 19)]
        for s in shapes:
            x = torch.Tensor(np.random.random_sample(s))
            # assert x2num.make_np(x, 'VID').shape[3] == 3

    def test_pytorch_write(self):
        with SummaryWriter() as w:
            w.add_scalar('scalar', torch.autograd.Variable(torch.rand(1)), 0)

    def test_pytorch_histogram(self):
        with SummaryWriter() as w:
            w.add_histogram('float histogram', torch.rand((50,)))
            w.add_histogram('int histogram', torch.randint(0, 100, (50,)))
