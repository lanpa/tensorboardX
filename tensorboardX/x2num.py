# DO NOT alter/distruct/free input object !

import numpy as np

def makenp(x, modality=None):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        if modality == 'IMG' and x.dtype == np.uint8:
            return x.astype(np.float32)/255.0
        return x
    if isinstance(x, float) or isinstance(x, int):
        return np.array([x])
    if 'torch' in str(type(x)):
        return pytorch_np(x, modality)

def pytorch_np(x, modality):
    import torch
    if isinstance(x, torch.autograd.variable.Variable):
        x = x.data
    x = x.cpu()
    assert isinstance(x, torch.Tensor), 'invalid input type'
    if modality == 'IMG':
        assert x.dim()<4 and x.dim()>1, 'input tensor should be 3D for color image or 2D for gray image.'
        if x.dim()==2:
            x = x.unsqueeze(0)
        x = x.permute(1,2,0) #CHW to HWC
    return x.numpy()


def theano_np(x):
    import theano
    pass

def caffe2_np(x):
    pass

def mxnet_np(x):
    pass

def chainer_np(x):
    import chainer
    #x = chainer.cuda.to_cpu(x.data)
    pass

