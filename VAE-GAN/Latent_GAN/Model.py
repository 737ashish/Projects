import torch
import torch.nn as nn

class Latent_Discriminator(nn.Module):
  def __init__(self, latent_size):
    super(Latent_Discriminator, self).__init__()
    self.disc = nn.Sequential(
      # Input: N x channels_img x 64 x 64
      nn.Conv2d(
        latent_size, latent_size*2, kernel_size=4, stride=2, padding=2
      ),
      nn.LeakyReLU(0.2),
      self._block(latent_size*2, latent_size*2, 4, 2, 2), # 16 x 16
      self._block(latent_size*2, latent_size*2, 4, 2, 2), # 8 x 8
      self._block(latent_size*2, latent_size*2, 4, 2, 2), # 4 x 4
      nn.Conv2d(latent_size*2, 1, kernel_size=4, stride=2, padding=2), # 1 x 1
            
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
      nn.BatchNorm2d(out_channels), #LayerNorm <==> Instance Norm
      nn.LeakyReLU(0.2),
      
    )

  def forward(self, x):
    return self.disc(x)

class Latent_Generator(nn.Module):
  def __init__(self, z_dim, latent_size):
    super(Latent_Generator, self).__init__()
    self.gen = nn.Sequential(
      # Input: N x z_dim x 1 x 1
      self._ct_block(z_dim, latent_size*2, 4, 1, 0), # N x f_g*16 x 4 x 4
      self._ct_block(latent_size*2, latent_size*2, 4, 1, 0), # 8 x 8
      self._ct_block(latent_size*2, latent_size*2, 4, 1, 0), #16 x 16
      self._c_block(latent_size*2, latent_size, 4, 2, 0), #32 x 32
      nn.Conv2d(
        latent_size, latent_size, kernel_size=4, stride=2, padding=0,
    ),
    nn.Tanh(), #[-1, 1] 

    )

  def _ct_block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
      ),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),

    )

  def _c_block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
      nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
      ),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),

    )


  def forward(self, x):
    return self.gen(x)