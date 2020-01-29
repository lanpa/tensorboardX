import unittest
import numpy as np
from tensorboardX import SummaryWriter

hparam = {'lr': [0.1, 0.01, 0.001],
          'bsize': [1, 2, 4],
          'n_hidden': [100, 200],
          'bn': [True, False]}

metrics = {'accuracy', 'loss'}

def train(lr, bsize, n_hidden):
    x = lr + bsize + n_hidden
    return x, x*5


class HparamsTest(unittest.TestCase):
    def test_smoke(self):
        i = 0
        with SummaryWriter() as w:
            for lr in hparam['lr']:
                for bsize in hparam['bsize']:
                    for n_hidden in hparam['n_hidden']:
                        for bn in hparam['bn']:
                            accu, loss = train(lr, bsize, n_hidden)
                            i = i + 1
                            w.add_hparams({'lr': lr, 'bsize': bsize, 'n_hidden': n_hidden, 'bn': bn},
                                            {'accuracy': accu, 'loss': loss}, name="trial"+str(i))

