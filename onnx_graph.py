import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import graph_onnx

dummy_input = Variable(torch.rand(8, 3, 224, 224))


from tensorboardX import SummaryWriter
with SummaryWriter() as w:
    model = torchvision.models.alexnet(pretrained=True)
    torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
    w.add_graph_onnx("test.proto")

with SummaryWriter() as w:
    model = torchvision.models.vgg19(pretrained=False)
    torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
    w.add_graph_onnx("test.proto")

with SummaryWriter() as w:
    model = torchvision.models.densenet121()
    torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
    w.add_graph_onnx("test.proto")

with SummaryWriter() as w:
    model = torchvision.models.resnet18()
    torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
    w.add_graph_onnx("test.proto")


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# not working
with SummaryWriter() as w:
    model = Net()
    dummy_input = Variable(torch.rand(8, 1, 28, 28))
    print(model(dummy_input))
    torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
    w.add_graph_onnx("test.proto")