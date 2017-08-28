import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)+F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

model = Mnist()

# if you want to show the input tensor, set requires_grad=True
res = model(torch.autograd.Variable(torch.Tensor(1,1,28,28), requires_grad=True))

writer = SummaryWriter()
writer.add_graph(model, res)

writer.close()
