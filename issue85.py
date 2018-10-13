import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class DummyModule(torch.nn.Module):
    def forward(self, x):
        V = Variable(torch.Tensor(2, 2))
        V[0, 0] = x
        # V = x
        return torch.sum(V * 3)


x = Variable(torch.Tensor([1]), requires_grad=True)
r = DummyModule()(x)
r.backward()
print(x.grad)

w = SummaryWriter()
x = Variable(torch.Tensor([1]), requires_grad=True)
w.add_graph(DummyModule(), x, verbose=True)