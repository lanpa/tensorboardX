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
    if modality == 'VID':
        x = _prepare_video(x)

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


def _prepare_video(V):

    b, c, t, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to power of 2
    while not is_power2(V.shape[0]):
        V = np.concatenate((V, np.zeros(shape=(1, c, t, h, w))), axis=0)

    b = V.shape[0]
    n_rows = 2**(int(np.log(b) / np.log(2)) // 2)
    n_cols = b // n_rows

    V = np.reshape(V, newshape=(n_rows, n_cols, c, t, h, w))
    V = np.transpose(V, axes=(3, 0, 4, 1, 5, 2))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V
