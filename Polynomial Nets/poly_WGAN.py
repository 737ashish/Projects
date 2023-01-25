import torch
import torch.nn as nn
from polynomial_nets import CP_L3

class Generator_CP_L3(nn.Module):
  def __init__(self, d, k, o):
    super(Generator_CP_L3, self).__init__()
    self.gen = CP_L3()
    self.layer_U1 = nn.Linear(d, k, bias=False)
    self.layer_U2 = nn.Linear(d, k, bias=False)
    self.layer_U3 = nn.Linear(d, k, bias=False)
    self.layer_C = nn.Linear(k, o)
   #[-1, 1] 


  def forward(self, z):
    out_U1 = self.layer_U1(z)
    out_U2 = self.layer_U2(z)
    out_U3 = self.layer_U3(z)
    out_U12 = out_U1*out_U2
    in_U3 = out_U1 + out_U12
    in_C = out_U3*in_U3 + in_U3
    x = self.layer_C(in_C)
    return x

class Poly_GeneratorC3(nn.Module):
  def __init__(self, d, k, o):
    super(Poly_GeneratorC3, self).__init__()
    self.layer_U1 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_U2 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_U3 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_C = nn.Conv2d(k, o, kernel_size=4, stride=2, padding=2, bias=True)
   #[-1, 1] 


  def forward(self, z):
    out_U1 = self.layer_U1(z)
    out_U2 = self.layer_U2(z)
    out_U3 = self.layer_U3(z)
    out_U12 = out_U1*out_U2
    in_U3 = out_U1 + out_U12
    in_C = out_U3*in_U3 + in_U3
    x = self.layer_C(in_C)
    return x