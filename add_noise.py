#!/usr/bin/env python
# coding: utf-8

# In[4]:


from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse


# In[6]:


BATCH_SIZE = 4


# In[ ]:


ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--dataset', type=str, 
    help='dataset to use'
)
args = vars(ap.parse_args())


# In[ ]:


if args['dataset'] == 'mnist' or args['dataset'] == 'fashionmnist':  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), 
    ])
    if args['dataset'] == 'mnist':
        trainset = datasets.MNIST(
            root='./data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    elif args['dataset'] == 'fashionmnist':
        trainset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
if args['dataset'] == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True, 
        transform=transform
    )
    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )


# In[7]:


trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE,
    shuffle=False
)


# In[8]:


def save_noisy_image(img, name):
    if img.size(1) == 3:
        img = img.view(img.size(0), 3, 32, 32)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 1, 28, 28)
        save_image(img, name)


# In[10]:


def poisson_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='poisson', clip=True))
        save_noisy_image(gauss_img, f"Images/{args['dataset']}_poisson.png")
        break


# In[ ]:


poisson_noise()

