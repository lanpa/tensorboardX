import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd.variable import Variable
import torch.nn.functional as F
from collections import OrderedDict
from tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import TensorDataset,DataLoader
from tensorboard.embedding import EmbeddingWriter
import os

#EMBEDDING VISUALIZATION FOR A TWO-CLASSES PROBLEM

#just a bunch of layers
class M(nn.Module):
    def __init__(self):
        super(M,self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3)
        self.cn2 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3)
        self.fc1 = nn.Linear(in_features=128,out_features=2)
    def forward(self,i):
        i = self.cn1(i)
        i = F.relu(i)
        i = F.max_pool2d(i,2)
        i =self.cn2(i)
        i = F.relu(i)
        i = F.max_pool2d(i,2)
        i = i.view(len(i),-1)
        i = self.fc1(i)
        i = F.log_softmax(i)
        return i

#get some random data around value
def get_data(value,shape):
    data= torch.ones(shape)*value
    #add some noise
    data += torch.randn(shape)**2
    return data

#dataset
#cat some data with different values
data = torch.cat((get_data(0,(100,1,14,14)),get_data(0.5,(100,1,14,14))),0)
#labels
labels = torch.cat((torch.zeros(100),torch.ones(100)),0)
#generator
gen = DataLoader(TensorDataset(data,labels),batch_size=25,shuffle=True)
#network
m = M()
#loss and optim
loss = torch.nn.NLLLoss()
optimizer = Adam(params=m.parameters())
#settings for train and log
num_epochs = 20
num_batches = len(gen)
embedding_log = 5
#WE NEED A WRITER! BECAUSE TB LOOK FOR IT!
writer_name = datetime.now().strftime('%B%d  %H:%M:%S')
writer = SummaryWriter(os.path.join("runs",writer_name))
#our brand new embwriter in the same dir
embedding_writer = EmbeddingWriter(os.path.join("runs",writer_name))
#TRAIN
for i in xrange(num_epochs):
    for j,sample in enumerate(gen):
        #reset grad
        m.zero_grad()
        optimizer.zero_grad()
        #get batch data
        data_batch = Variable(sample[0],requires_grad=True).float()
        label_batch = Variable(sample[1],requires_grad=False).long()
        #FORWARD
        out = m(data_batch)
        loss_value = loss(out,label_batch)
        #BACKWARD
        loss_value.backward()
        optimizer.step()
        #LOGGING
        if j % embedding_log == 0:
            print("loss_value:{}".format(loss_value.data[0]))
            #we need 3 dimension for tensor to visualize it!
            out = torch.cat((out,torch.ones(len(out),1)),1)
            #write the embedding for the timestep
            embedding_writer.add_embedding(out.data,metadata=label_batch.data,label_img=data_batch.data,timestep=(i*num_batches)+j)

writer.close()

#tensorboard --logdir runs
#you should now see a dropdown list with all the timestep, latest timestep should have a visible separation between the two classes