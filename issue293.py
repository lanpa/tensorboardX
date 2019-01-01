import torch
import torchvision.models as models
from tensorboardX import SummaryWriter

device = 'cpu'

net = torch.nn.DataParallel(torch.nn.DataParallel(models.__dict__['resnet50']().to(device)))
# print(type(net))
# print(net)
# torch.save(net, "88888.pth")
# net = models.__dict__['resnet50']()
dump_input = torch.rand((10, 3, 224, 224))
with SummaryWriter() as w:
    w.add_graph(net, dump_input, verbose=True)

