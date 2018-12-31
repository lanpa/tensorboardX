from tensorboardX import summary

import numpy as np
import pytest
import unittest
from tensorboardX.utils import make_grid

class UtilsTest(unittest.TestCase):
    def test_to_HWC(self):
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'chw')
        assert converted.shape == (32, 32, 3)
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'nchw')
        assert converted.shape == (64, 256, 3)
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'hw')
        assert converted.shape == (32, 32, 3)

def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    assert(len(set(input_format)) == len(input_format)), "You can not use the same dimension shordhand twice."
    assert(len(tensor.shape) == len(input_format)), "size of input tensor and input format are different"
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in 'NCHW']
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in 'HWC']
        return tensor.transpose(index)

    if len(input_format) == 2:
        index = [input_format.find(c) for c in 'HW']
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor
