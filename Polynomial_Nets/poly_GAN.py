import torch
import torch.nn as nn
from polynomial_nets import CP_L3, CP_C3, NCP_L3, CP_CT3, Chebyshev_L3, Attention

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, L = real.shape
    epsilon = torch.randn((BATCH_SIZE, 1)).repeat(1, L).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    #calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm -1) ** 2)
    return gradient_penalty

def gradient_penalty_C(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.randn((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    #calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm -1) ** 2)
    return gradient_penalty


class Generator_attention(nn.Module):
  def __init__(self, z_dim, output, rank):
    super(Generator_attention, self).__init__()
    self.gen = nn.Sequential(Attention(z_dim, rank, output), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)

class Generator_Cheby_L3(nn.Module):
  def __init__(self, z_dim, output, rank):
    super(Generator_Cheby_L3, self).__init__()
    self.gen = nn.Sequential(Chebyshev_L3(z_dim, rank, output), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)



class Generator_CP_L3(nn.Module):
  def __init__(self, z_dim, output, rank):
    super(Generator_CP_L3, self).__init__()
    self.gen = nn.Sequential(CP_L3(z_dim, rank, output), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)

class Generator_CP_CT3(nn.Module):
  def __init__(self, z_dim, output, rank):
    super(Generator_CP_CT3, self).__init__()
    self.gen = nn.Sequential(CP_CT3(z_dim, rank, output), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)

class Generator_CP_C3(nn.Module):
  def __init__(self, z_dim, output, rank):
    super(Generator_CP_C3, self).__init__()
    self.gen = nn.Sequential(CP_C3(z_dim, rank, output), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)

class Generator_NCP_L3(nn.Module):
  def __init__(self, z_dim, output, rank, LHP):
    super(Generator_NCP_L3, self).__init__()
    self.gen = nn.Sequential(NCP_L3(z_dim, rank, output, LHP), nn.Tanh())


  def forward(self, z):
    return  self.gen(z)

class Critic_CP_L3(nn.Module):
  def __init__(self, input, rank):
    super(Critic_CP_L3, self).__init__()
    self.critic = CP_L3(input, rank, 1)

  def forward(self, x):
    return self.critic(x)

class Discriminator_CP_L3(nn.Module):
  def __init__(self, input, rank):
    super(Discriminator_CP_L3, self).__init__()
    self.disc = nn.Sequential(CP_L3(input, rank, 1), nn.Sigmoid)

  def forward(self, x):
    return self.disc(x)
    

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

class Discriminator32(nn.Module):
  def __init__(self, channels_img, features_d):
    super(Discriminator32, self).__init__()
    self.disc = nn.Sequential(
      # Input: N x channels_img x 64 x 64
      nn.Conv2d(
        channels_img, features_d, kernel_size=4, stride=2, padding=1
      ),
      nn.LeakyReLU(0.2),
      self._block(features_d, features_d*2, 3, 1, 1), # 16 x 16
      self._block(features_d*2, features_d*4, 4, 2, 1), # 4 x 4
      nn.Conv2d(features_d*4, 1, kernel_size=4, stride=2, padding=1), # 1 x 1
            
    )

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
      nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
      ),
      nn.InstanceNorm2d(out_channels, affine=True), #LayerNorm <==> Instance Norm
      nn.LeakyReLU(0.2),
      
    )

  def forward(self, x):
    return self.disc(x)

class Generator32(nn.Module):
  def __init__(self, z_dim, channels_img, features_g):
    super(Generator32, self).__init__()
    self.gen = nn.Sequential(
      # Input: N x z_dim x 1 x 1
      self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
      self._block(features_g*16, features_g*8, 4, 2, 1), # 8 x 8
      self._block(features_g*8, features_g*2, 4, 2, 1), #32 x 32
      nn.ConvTranspose2d(
        features_g*2, channels_img, kernel_size=4, stride=2, padding=1,
    ),
    nn.Tanh(), #[-1, 1] 

    )

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
      ),
      #nn.BatchNorm2d(out_channels),
      nn.ReLU(),

    )

  def forward(self, x):
    return self.gen(x)