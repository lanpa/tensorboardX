import torch
from torch import nn
import torch.nn.functional as F
from torchvision import utils
import numpy as np

from tensorboardX import SummaryWriter

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
        return F.log_softmax(x, dim=-1)

def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)

    return grid

net = Net()

writer = SummaryWriter()

for name, param in net.named_parameters(): 
    #writer.add_histogram(tag=name, values=param.data, global_step=1)
    w_img = param.data
    shape = list(w_img.size())
    
    if len(shape) == 1:
        w_img = w_img.view(shape[0], 1, 1, 1)
        if shape[0] > 16:
            nrow = 16
        else:
            nrow = shape[0]
        grid = vistensor(w_img, ch=0, allkernels=True, padding=0, nrow=nrow)

    elif len(shape) == 2:
        if shape[0] > shape[1]:
            w_img = torch.transpose(w_img, 0, 1)
            shape = list(w_img.size())
        w_img = w_img.view(shape[0] * shape[1], 1, 1, 1)
        grid = vistensor(w_img, ch=0, allkernels=True,
                                nrow=int(np.sqrt(shape[0] * shape[1])), padding=0)

    elif len(shape) == 3:
        w_img = w_img.view(shape[0], 1, shape[1], shape[2])
        grid = vistensor(w_img, ch=0, allkernels=True, padding=1)

    elif len(shape) == 4:
        grid = vistensor(w_img, ch=0, allkernels=True, padding=1)

    else:
        continue

    writer.add_histogram(tag=name, values=param.data, global_step=1)
    writer.add_image('img_'+name, grid.numpy(), 1)

exit()
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import utils
import numpy as np

from tensorboardX import SummaryWriter

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
        return F.log_softmax(x, dim=-1)

def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)

    return grid

net = Net()

writer = SummaryWriter()

counter = 1
for name, param in net.named_parameters(): 
    print(param.data.shape)
    writer.add_histogram(tag=name, values=param, global_step=1)
    w_img = param.data
    shape = list(w_img.size())
    
    if len(shape) == 1:
        w_img = w_img.view(shape[0], 1, 1, 1)
        if shape[0] > 16:
            nrow = 16
        else:
            nrow = shape[0]
        grid = vistensor(w_img, ch=0, allkernels=True, padding=0, nrow=nrow)

    elif len(shape) == 2:
        if shape[0] > shape[1]:
            w_img = torch.transpose(w_img, 0, 1)
            shape = list(w_img.size())
        w_img = w_img.view(shape[0] * shape[1], 1, 1, 1)
        grid = vistensor(w_img, ch=0, allkernels=True,
                                nrow=int(np.sqrt(shape[0] * shape[1])), padding=0)

    elif len(shape) == 3:
        w_img = w_img.view(shape[0], 1, shape[1], shape[2])
        grid = vistensor(w_img, ch=0, allkernels=True, padding=1)

    elif len(shape) == 4:
        grid = vistensor(w_img, ch=0, allkernels=True, padding=1)

    else:
        continue

    # writer.add_histogram(tag=name, values=param.data.numpy(), global_step=counter)
    # writer.add_histogram(name, param, 1, bins='sqrt')
    # grid = grid.unsqueeze(0)
    # print(grid.shape)
    writer.add_image(name, grid, 1)
    # writer.add_image('name', grid, counter)

    print('hihi')
    counter +=1