from tensorboardX import SummaryWriter
import time



hparam = {'lr': [0.1, 0.01, 0.001],
          'bsize': [1, 2, 4],
          'n_hidden': [100, 200]}

metrics = {'accuracy', 'loss'}


for i in range(5):
    with SummaryWriter() as w_hparam:
        w_hparam.add_hparams({'lr': 0.1*i, 'bsize': i},
                {'accuracy': 10*i, 'loss': 10*i})
        w_hparam.close()
    time.sleep(2)
