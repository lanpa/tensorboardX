from tensorboardX import summary

import numpy as np


def test_uint8_image():
    '''
    Tests that uint8 image (pixel values in [0, 255]) is not changed
    '''
    test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
    scale_factor = summary._calc_scale_factor(test_image)
    assert scale_factor == 1, 'Values are already in [0, 255], scale factor should be 1'


def test_float32_image():
    '''
    Tests that float32 image (pixel values in [0, 1]) are scaled correctly
    to [0, 255]
    '''
    test_image = np.random.rand(3, 32, 32).astype(np.float32)
    scale_factor = summary._calc_scale_factor(test_image)
    assert scale_factor == 255, 'Values are in [0, 1], scale factor should be 255'
