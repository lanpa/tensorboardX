from tensorboardX import x2num
import torch
import numpy as np
shapes = [(3, 10, 10), (1, ), (1, 2, 3, 4, 5)]

def test_pytorch_np():

    for shape in shapes:
        # regular tensor
        assert(isinstance(x2num.makenp(torch.Tensor(*shape)), np.ndarray))

        # CUDA tensor
        assert(isinstance(x2num.makenp(torch.Tensor(*shape).cuda()), np.ndarray))

        # regular variable
        assert(isinstance(x2num.makenp(torch.autograd.variable.Variable(torch.Tensor((*shape)))), np.ndarray))

        # CUDA variable
        assert(isinstance(x2num.makenp(torch.autograd.variable.Variable(torch.Tensor((*shape))).cuda()), np.ndarray))

    # python primitive type
    assert(isinstance(x2num.makenp(0), np.ndarray))
    assert(isinstance(x2num.makenp(0.1), np.ndarray))

