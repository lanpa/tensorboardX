from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from tensorboardX import x2num


class NumpyTest(unittest.TestCase):
    def test_scalar(self):
        res = x2num.make_np(1.1)
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(1000000000000000000000)
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float16(1.00000087))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float128(1.00008 + 9))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.int64(100000000000))
        assert isinstance(res, np.ndarray) and res.shape == (1,)

    def test_make_grid(self):
        pass
