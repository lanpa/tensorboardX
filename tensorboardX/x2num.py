# DO NOT alter/distruct/free input object !

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_nan(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        logger.warning('NaN or Inf found in input tensor.')
    return array


def make_np(x):
    if isinstance(x, list):
        return check_nan(np.array(x))
    if isinstance(x, np.ndarray):
        return check_nan(x)
    if np.isscalar(x):
        return check_nan(np.array([x]))
    if 'torch' in str(type(x)):
        return check_nan(prepare_pytorch(x))
    if 'chainer' in str(type(x)):
        return check_nan(prepare_chainer(x))
    if 'mxnet' in str(type(x)):
        return check_nan(prepare_mxnet(x))
    if 'jax' in str(type(x)):
        return check_nan(np.array(x))
    if 'paddle' in str(type(x)):
        return check_nan(np.array(x))
    raise NotImplementedError(
        f'Got {type(x)}, but expected numpy array or torch tensor.')


def prepare_pytorch(x):
    import torch
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x


def prepare_theano(x):
    import theano
    pass


def prepare_mxnet(x):
    x = x.asnumpy()
    return x


def prepare_chainer(x):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    return x
