from tensorboardX import x2num
import torch
import numpy as np
tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]

def test_pytorch_np():

    for tensor in tensors:
        # regular tensor
        assert isinstance(x2num.makenp(tensor), np.ndarray)

        # CUDA tensor
        if torch.cuda.device_count()>0:
            assert isinstance(x2num.makenp(tensor.cuda()), np.ndarray)

        # regular variable
        assert isinstance(x2num.makenp(torch.autograd.variable.Variable(tensor)), np.ndarray)

        # CUDA variable
        if torch.cuda.device_count()>0:
            assert isinstance(x2num.makenp(torch.autograd.variable.Variable(tensor)).cuda(), np.ndarray)

    # python primitive type
    assert(isinstance(x2num.makenp(0), np.ndarray))
    assert(isinstance(x2num.makenp(0.1), np.ndarray))

shapes = [(77, 3, 13, 7), (77, 1, 13, 7), (3, 13, 7), (1, 13, 7), (13, 7)]
def test_pytorch_img():
    for s in shapes:
        x = torch.Tensor(np.random.random_sample(s))
        assert x2num.makenp(x, 'IMG').shape[2]==3