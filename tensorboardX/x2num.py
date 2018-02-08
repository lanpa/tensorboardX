# DO NOT alter/distruct/free input object !

import numpy as np


def makenp(x, modality=None):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        if modality == 'IMG' and x.dtype == np.uint8:
            return x.astype(np.float32) / 255.0
        return x
    if np.isscalar(x):
        return np.array([x])
    if 'torch' in str(type(x)):
        return pytorch_np(x, modality)
    if 'chainer' in str(type(x)):
        return chainer_np(x, modality)
    if 'mxnet' in str(type(x)):
        return mxnet_np(x, modality)


def pytorch_np(x, modality):
    import torch
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def theano_np(x):
    import theano
    pass


def caffe2_np(x):
    pass


def mxnet_np(x, modality):
    x = x.asnumpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def chainer_np(x, modality):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def make_grid(I, ncols=8):
    assert isinstance(I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim == 4 and I.shape[1] == 3
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, H * nrows, W * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas


def _prepare_image(I):
    assert isinstance(I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim == 2 or I.ndim == 3 or I.ndim == 4
    if I.ndim == 4:  # NCHW
        if I.shape[1] == 1:  # N1HW
            I = np.concatenate((I, I, I), 1)  # N3HW
        assert I.shape[1] == 3
        I = make_grid(I)  # 3xHxW
    if I.ndim == 3 and I.shape[0] == 1:  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    if I.ndim == 2:  # HxW
        I = np.expand_dims(I, 0)  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    I = I.transpose(1, 2, 0)

    return I
