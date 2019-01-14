from tensorboardX import summary

import numpy as np
import pytest
import unittest


class SummaryTest(unittest.TestCase):
    def test_uint8_image(self):
        '''
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        '''
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 1, 'Values are already in [0, 255], scale factor should be 1'

    def test_float32_image(self):
        '''
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        '''
        test_image = np.random.rand(3, 32, 32).astype(np.float32)
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 255, 'Values are in [0, 1], scale factor should be 255'

    def test_list_input(self):
        with pytest.raises(Exception) as e_info:
            summary.histogram('dummy', [1,3,4,5,6], 'tensorflow')

    def test_empty_input(self):
        print('expect error here:')
        with pytest.raises(Exception) as e_info:
            summary.histogram('dummy', np.ndarray(0), 'tensorflow')

    def test_image_with_boxes(self):
        summary.image_boxes('dummy',
                            np.random.rand(3, 32, 32).astype(np.float32),
                            np.array([[10, 10, 40, 40]]))

    def test_image_with_one_channel(self):
        summary.image('dummy', np.random.rand(1, 32, 32).astype(np.float32), dataformats='CHW')
        summary.image('dummy', np.random.rand(4, 1, 32, 32).astype(np.float32), dataformats='NCHW')
        summary.image('dummy', np.random.rand(4, 3, 32, 32).astype(np.float32), dataformats='NCHW')
        summary.image('dummy', np.random.rand(32, 32).astype(np.float32), dataformats='HW')

    def test_video(self):
        summary.video('dummy', np.random.rand(16, 48, 1, 28, 28).astype(np.float32))

    def test_audio(self):
        summary.audio('dummy', np.random.rand(65535))

    def test_text(self):
        summary.text('dummy', 'text 123')

    def test_histogram(self):
        summary.histogram('dummy', np.random.rand(1024), bins='auto', max_bins=5)
        summary.histogram('dummy', np.random.rand(1024), bins='fd', max_bins=5)
        summary.histogram('dummy', np.random.rand(1024), bins='doane', max_bins=5)
