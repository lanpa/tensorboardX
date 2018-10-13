import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # 实际上并不成立
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)  # 通过此操作，feature map的尺寸变为原来的一半
        downrelu = nn.LeakyReLU(0.2, True)  # 大于0时为0，小于0时乘上一个斜率
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:  # 如果是U-NET网络结构中的最外层部分
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 如果是U-NET网络结构中的最低部
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)  # size变为原来的了两倍
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # U-NET网络结构的中间部分
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,  # 因为concate的原因，input channel的数目变为原来的两倍
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)  # 在第一个维度上将x与model(x)进行cat一起，向上up sampling

import torch.nn as nn
if __name__ =="__main__":
    writer = SummaryWriter(comment='unet-skip')
    dummy_input = Variable(torch.rand(13, 512, 2, 2))
    #net = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False)
    norm_layer = nn.BatchNorm2d
    ngf = 64
    net2 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
    with writer:
        writer.add_graph(net2, (dummy_input,))