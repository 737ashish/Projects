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
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img_list = os.listdir('/home/ramana44/pytorch-vae/anime_data/set1')
# The above address consists of all the images
#img_list.extend(os.listdir('/home/ramana44/pytorch-vae/anime_data/set2'))
#print('len(img_list)', len(img_list))

transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor()
])

anime_data = datasets.ImageFolder('/home/ramana44/pytorch-vae/anime_data', transform = transform)

#print(anime_data.classes)
#print(len(anime_data))

#Considering 75% of data for training and remaining for testing
train_size = int(len(img_list)*0.75)
test_size = len(img_list) - train_size

train_set, test_set = torch.utils.data.random_split(anime_data, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

testiter = iter(test_loader)
images_check, labels_check = testiter.next()


model = VAE(image_channels=images_check.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

epochs = 100

for epoch in range(epochs):
    for idx, (images, label) in enumerate(train_loader):
        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss', loss)
    print("Epoch : ", epoch)


torch.save(model.state_dict(), '/home/ramana44/pytorch-vae/saved_models/animesCnnVae.torch')