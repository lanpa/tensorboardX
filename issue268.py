import torch
import torchvision
from tensorboardX import SummaryWriter

model = torchvision.models.vgg19()
dummy_input = torch.rand(1, 3, 224, 224)
dummy_state = model.forward(dummy_input)




with SummaryWriter(comment='SalGANplus') as w:
    #model = torchvision.models.vgg19()
    w.add_graph(model, (dummy_input, ), verbose=False)

