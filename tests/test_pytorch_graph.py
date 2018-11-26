import torch
from tensorboardX import SummaryWriter

dummy_input = (torch.zeros(1, 3),)


class myLinear(torch.nn.Module):
    def __init__(self):
        super(myLinear, self).__init__()
        self.l = torch.nn.Linear(3, 5)
    def forward(self, x):
        return self.l(x)

with SummaryWriter(comment='LinearModel') as w:
    w.add_graph(myLinear(), dummy_input, True)