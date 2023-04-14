import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'selu':
        layer = nn.SELU()
    elif act_type == 'elu':
        layer = nn.ELU()
    elif act_type == 'silu':
        layer = nn.SiLU()
    elif act_type == 'rrelu':
        layer = nn.RReLU()
    elif act_type == 'celu':
        layer = nn.CELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
    
#lfe
class lfe(nn.Module):
  def __init__(self, in_c= 3, out_c=64):
    super(lfe, self).__init__()
    self.conv1= conv_block(in_c, out_c, kernel_size= 9)
    self.conv2 = conv_block(out_c, out_c, kernel_size= 1)
  def forward(self, x):
    x1= self.conv2(self.conv1(x))
    return x1
#rec
class rec(nn.Module):
  def __init__(self, in_c=64, out_c= 3):
    super(rec, self).__init__()
    self.conv1= conv_block(in_c, 64, kernel_size= 3)
    self.conv2= nn.Conv2d(64, 3, kernel_size= 3)
    self.TanH= nn.Tanh()
  def forward(self,x):
    x= self.TanH(F.pad(self.conv2(self.conv1(x)),(1,1,1,1),"constant"))
    return x
class upscale(nn.Module):
    def __init__(self, in_c=64, out_c= 64):
        super(upscale, self).__init__()
        self.conv1= nn.Conv2d(64, 128, kernel_size=3)
        self.Conv2= conv_block(64,32,kernel_size= 3, norm_type = 'batch')
        self.pixelShuffle= nn.PixelShuffle(2)
        self.prelu= nn.PReLU()
        self.batchNorm=  nn.BatchNorm2d(32)
    def forward(self, x):
        xl =self.conv1(x)
        xl= self.pixelShuffle(xl)
        x1= self.batchNorm(xl)
        xl = self.prelu(xl)
        xl=F.pad(xl,(2,2,2,2))
        xr=  F.interpolate(x,scale_factor=2,mode= 'nearest')
        xr= self.Conv2(xr)
        out= torch.cat((xr, xl),1)
        return out
class Ca(nn.Module):
 def __init__(self, in_c=64):
    super(Ca, self).__init__()
    self.AvgPool= nn.AvgPool2d((1,1))
    self.conv1= conv_block(in_c, 16, kernel_size=1)
    self.conv2=nn.Conv2d(16, 64, kernel_size= 1)
    self.act= nn.Sigmoid()
 def forward(self, x):
    x1= self.AvgPool(x)
    x1= self.conv1(x1)
    x1= self.conv2(x1)
    x1= x*x1
    return x1
class resBlock(nn.Module):
 def __init__(self, in_c=64, out_c= 64):
    super(resBlock, self).__init__()
    self.convl1= conv_block(in_c, out_c, kernel_size= 3)
    self.convl2= conv_block(in_c, out_c, kernel_size= 1)
    self.convl3= conv_block(in_c, out_c, kernel_size= 1)
    self.convm1= conv_block(in_c, out_c, kernel_size= 3)
    self.convm2= conv_block(in_c, out_c, kernel_size= 1)
    self.convm3= conv_block(in_c, out_c, kernel_size= 1)
    self.convr1= conv_block(in_c, out_c, kernel_size= 3)
    self.convr2= conv_block(in_c, out_c, kernel_size= 1)
    self.convr3= conv_block(in_c, out_c, kernel_size= 1)
    self.ca1= Ca()
    self.ca2= Ca()
    self.ca3= Ca()
    self.conv= conv_block(in_c*3, 64, kernel_size=1)
 def forward(self, x):
    xl= self.convl1(x)
    xl= self.convl2(xl)
    xl= self.convl3(xl)
    xl = self.ca1.forward(xl)
    xm = self.convm1(x)
    xm= self.convm2(xm)
    xm= self.convm3(xm)
    xm= self.ca2.forward(xm)
    xr = self.convr1(x)
    xr= self.convr2(xr)
    xr= self.convr3(xr)
    xr = self.ca3.forward(xr)
    out = torch.cat((xl,xm,xr), axis=1)
    out= self.conv(out)
    return out+x

#hfe
class HFE(nn.Module):
  def __init__(self, in_c=64, out_c= 64):
    super(HFE, self).__init__()
    self.resBlock1= resBlock()
    self.resBlock2= resBlock()
    self.resBlock3= resBlock()
    self.resBlock4= resBlock()
    self.upscale1= upscale()
    self.resBlock5= resBlock()
    self.resBlock6= resBlock()
    self.upscale2= upscale()
    self.resBlock7= resBlock()
    self.resBlock8= resBlock()
    self.upscale3= upscale()
  def forward(self, x):
    x1= self.resBlock1(x)
    x1= self.resBlock2(x1)
    x1= self.resBlock3(x1)
    x1= self.resBlock4(x1)
    x1= self.upscale1(x1)
    x1= self.resBlock5(x1)
    x1= self.resBlock6(x1)
    x1= self.upscale2(x1)
    x1= self.resBlock7(x1)
    x1= self.resBlock8(x1)
    x1= self.upscale3(x1)
    return x1

#grln 
class GRLN(nn.Module):
  def __init__(self, in_c=64, out_c= 64):
    super(GRLN, self).__init__()
    self.conv1= conv_block(3,out_c, kernel_size= 1)
    self.conv2= conv_block(in_c, 16, kernel_size= 1)
    self.conv3= nn.Conv2d(16, 3, kernel_size= 1)
    self.act= nn.Tanh()
  def forward(self, x):
    x1= F.interpolate(x, scale_factor=8, mode='bicubic')
    x1= self.conv1(x1)
    x1= self.conv2(x1)
    x1= self.conv3(x1)
    x1= self.act(x1)
    return x1

class myNetwork(nn.Module):
    def __init__(self, in_c=64, out_c= 64):
        super(myNetwork, self).__init__()
        self.lfe= lfe()
        self.hfe= HFE()
        self.grln= GRLN()
        self.rec= rec()
    def forward(self, x):
        x1= self.lfe(x)
        x1= self.hfe(x1)
        x1= self.rec(x1)
        x2= self.grln(x)
        return x1+x2