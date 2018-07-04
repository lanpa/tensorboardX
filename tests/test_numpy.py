from tensorboardX import x2num
import numpy as np


def test_scalar():
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


def test_make_grid():
    pass
