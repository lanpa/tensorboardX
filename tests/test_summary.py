from __future__ import absolute_import, division, print_function, unicode_literals
from tensorboardX import summary
from .expect_reader import compare_proto, write_proto
import numpy as np
import pytest
import unittest
# compare_proto = write_proto  # massive update expect

def tensor_N(shape, dtype=float):
    numel = np.prod(shape)
    x = (np.arange(numel, dtype=dtype)).reshape(shape)
    return x

class SummaryTest(unittest.TestCase):
    def test_uint8_image(self):
        '''
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        '''
        test_image = tensor_N(shape=(3, 32, 32), dtype=np.uint8)
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 1, 'Values are already in [0, 255], scale factor should be 1'

    def test_float32_image(self):
        '''
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        '''
        test_image = tensor_N(shape=(3, 32, 32))
        scale_factor = summary._calc_scale_factor(test_image)
        assert scale_factor == 255, 'Values are in [0, 1], scale factor should be 255'

    def test_list_input(self):
        with pytest.raises(Exception):
            summary.histogram('dummy', [1, 3, 4, 5, 6], 'tensorflow')

    def test_empty_input(self):
        print('expect error here:')
        with pytest.raises(Exception):
            summary.histogram('dummy', np.ndarray(0), 'tensorflow')

    def test_image_with_boxes(self):
        compare_proto(summary.image_boxes('dummy',
                            tensor_N(shape=(3, 32, 32)),
                            np.array([[10, 10, 40, 40]])), self)

    def test_image_with_one_channel(self):
        compare_proto(summary.image('dummy', tensor_N(shape=(1, 8, 8)), dataformats='CHW'), self)

    def test_image_with_one_channel_batched(self):
        compare_proto(summary.image('dummy', tensor_N(shape=(2, 1, 8, 8)), dataformats='NCHW'), self)

    def test_image_with_3_channel_batched(self):
        compare_proto(summary.image('dummy', tensor_N(shape=(2, 3, 8, 8)), dataformats='NCHW'), self)

    def test_image_without_channel(self):
        compare_proto(summary.image('dummy', tensor_N(shape=(8, 8)), dataformats='HW'), self)

    def test_video(self):
        try:
            import moviepy
        except ImportError:
            return
        compare_proto(summary.video('dummy', tensor_N(shape=(4, 3, 1, 8, 8))), self)
        summary.video('dummy', tensor_N(shape=(16, 48, 1, 28, 28)))
        summary.video('dummy', tensor_N(shape=(20, 7, 1, 8, 8)))

    def test_audio(self):
        compare_proto(summary.audio('dummy', tensor_N(shape=(42,))), self)

    def test_text(self):
        compare_proto(summary.text('dummy', 'text 123'), self)

    def test_histogram_auto(self):
        compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='auto', max_bins=5), self)

    def test_histogram_fd(self):
        compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='fd', max_bins=5), self)

    def test_histogram_doane(self):
        compare_proto(summary.histogram('dummy', tensor_N(shape=(1024,)), bins='doane', max_bins=5), self)
