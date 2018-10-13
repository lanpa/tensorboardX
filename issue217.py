import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter()
for i in range(100):
  v = np.random.uniform(0, 5, (64,3,7,7))
  writer.add_histogram('x', v, i, bins='sqrt')
#   writer.add_histogram('x', v, 2)
writer.close()
