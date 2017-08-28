import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
#import skimage
#from skimage import data, io

resnet18 = models.resnet18(False)
writer = SummaryWriter()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

for n_iter in range(100):
    s1 = torch.rand(1) # value to keep
    s2 = torch.rand(1)
    writer.add_scalar('data/scalar1', s1[0], n_iter) # data grouping by `slash`
    writer.add_scalar('data/scalar2', s2, n_iter) # passing Tensor is OK!
    x = torch.rand(32, 3, 64, 64) # output from network
    if n_iter%10==0:
        x = vutils.make_grid(x, normalize=True, scale_each=True)   
        writer.add_image('Image', x, n_iter) # Tensor
        #writer.add_image('astronaut', skimage.data.astronaut(), n_iter) # numpy
        #writer.add_image('imread', skimage.io.imread('screenshots/audio.png'), n_iter) # numpy
        x = torch.zeros(sample_rate*2)
        for i in range(x.size(0)):
            x[i] = np.cos(freqs[n_iter//10]*np.pi*float(i)/float(sample_rate)) # sound amplitude should in [-1, 1]
        writer.add_audio('myAudio', x, n_iter)
        writer.add_text('Text', 'text logged at step:'+str(n_iter), n_iter)
        writer.add_text('another Text', 'another text logged at step:'+str(n_iter), n_iter)
        for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param, n_iter)

dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))        
writer.close()