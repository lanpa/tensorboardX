from tensorboardX import SummaryWriter
import time
import random


hparam = {'lr': [0.1, 0.01, 0.001],
          'bsize': [1, 2, 4],
          'n_hidden': [100, 200],
          'bn': [True, False]}

metrics = {'accuracy', 'loss'}

def train(lr, bsize, n_hidden):
    x = random.random()
    return x, x*5

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

