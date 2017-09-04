from tensorboardX import x2num
import chainer
import numpy as np
chainer.Variable
tensors = [chainer.Variable(np.random.rand(3, 10, 10)),
             chainer.Variable(np.random.rand(1)),
              chainer.Variable(np.random.rand(1, 2, 3, 4, 5))]

def test_chainer_np():

    for tensor in tensors:
        # regular variable
        assert isinstance(x2num.makenp(tensor), np.ndarray)

    # python primitive type
    assert(isinstance(x2num.makenp(0), np.ndarray))
    assert(isinstance(x2num.makenp(0.1), np.ndarray))

shapes = [(77, 3, 13, 7), (77, 1, 13, 7), (3, 13, 7), (1, 13, 7), (13, 7)]
def test_chainer_img():
    for s in shapes:
        x = chainer.Variable(np.random.random_sample(s))
        assert x2num.makenp(x, 'IMG').shape[2]==3