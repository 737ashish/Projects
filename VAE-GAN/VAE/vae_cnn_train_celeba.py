import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from pushover import notify
from utils import makegif
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display

from vae import Flatten, UnFlatten, VAE

#%load_ext autoreload
#%autoreload 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = 32

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.Resize(64),transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.Resize(64),transforms.ToTensor()])) 

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)

fixed_x, _ = next(iter(train_loader))
image_channels = 3

print('image_channels', image_channels)

model = VAE(image_channels=image_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


batch_size_cfs = 50
#image_batches_trn = torch.load('/home/ramana44/pytorch-vae/celebA_mustache1000/mustache1000.pt').to(device)

image_batches_trn = torch.load('/home/ramana44/pytorch-vae/noMustache20000Images/noMustache20000.pt').to(device)


image_batches_trn = image_batches_trn.reshape(int(image_batches_trn.shape[0]/batch_size_cfs), batch_size_cfs, 3, 64,64)


print('image_batches_trn.shape', image_batches_trn.shape)


epochs = 1000

for epoch in range(epochs):
    #for idx, (images, _) in enumerate(train_loader):
    inum = 0
    for images in image_batches_trn:    
        inum = inum+1
        #images = torch.cuda.FloatTensor(images)
        #print('images.shape', images.shape)
        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                #epochs, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
        #print(to_print)
    print('loss', loss)
    print("Epoch : ", epoch)

# notify to android when finished training
#notify(to_print, priority=1)

torch.save(model.state_dict(), '/home/ramana44/pytorch-vae/saved_models/celeba20000NoMustacheCnnVae.torch')