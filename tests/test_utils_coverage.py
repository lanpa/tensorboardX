import unittest
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import utils

class UtilsCoverageTest(unittest.TestCase):
    def test_figure_to_image_single(self):
        fig = plt.figure()
        plt.plot([1, 2], [1, 2])
        img = utils.figure_to_image(fig)
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(len(img.shape), 3) # CHW
        self.assertEqual(img.shape[0], 3) # RGB

    def test_figure_to_image_list(self):
        fig1 = plt.figure()
        plt.plot([1, 2], [1, 2])
        fig2 = plt.figure()
        plt.plot([2, 1], [2, 1])
        imgs = utils.figure_to_image([fig1, fig2])
        self.assertTrue(isinstance(imgs, np.ndarray))
        self.assertEqual(len(imgs.shape), 4) # NCHW
        self.assertEqual(imgs.shape[0], 2)
        self.assertEqual(imgs.shape[1], 3)

    def test_prepare_video_non_power2(self):
        # Test padding logic for non-power-of-2 batch size
        V = np.zeros((3, 5, 3, 10, 10), dtype=np.uint8)
        V_out = utils._prepare_video(V)
        # 3 should be padded to 4. 
        # Output shape: (T, n_rows * H, n_cols * W, C)
        # b = 3. b.bit_length() = 2.
        # n_rows = 2**((2 - 1) // 2) = 2**0 = 1
        # n_cols = 4 // 1 = 4
        self.assertEqual(V_out.shape, (5, 10, 40, 3))

    def test_prepare_video_uint8(self):
        V = np.ones((1, 1, 1, 1, 1), dtype=np.uint8) * 255
        V_out = utils._prepare_video(V)
        self.assertEqual(V_out[0, 0, 0, 0], 1.0)
