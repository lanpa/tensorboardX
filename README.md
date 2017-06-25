# tensorboard-pytorch

Write tensorboard events with simple command.

## install

`#tested on anaconda3`

`pip install tensorboard-pytorch`

`pip install tensorflow`


## usage
```python
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from datetime import datetime
from tensorboard import SummaryWriter
resnet18 = models.resnet18()
writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
for n_iter in range(100):
    M_global = torch.rand(1) # value to keep
    writer.add_scalar('M_global', M_global[0], n_iter)
    x = torch.rand(32, 3, 64, 64) # output from network
    if n_iter%10==0:
        x = vutils.make_grid(x, normalize=True, scale_each=True)   
        writer.add_image('Image', x, n_iter)
        for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
writer.close()
```

` tensorboard --logdir runs`  

## tweaks
To show more images in tensorboard's image tab, just
modify the hardcoded `event_accumulator` in 
`~/anaconda3/lib/python3.6/site-packages/tensorflow/tensorboard/backend/application.py`
as you wish.


## TODO
push pytorch specific utilities.

## reference:

https://github.com/TeamHG-Memex/tensorboard_logger

https://github.com/dmlc/tensorboard
