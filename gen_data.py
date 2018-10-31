from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import struct

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=False, **kwargs)



def train():
    with open(target_dir + 'MNIST_train.bin', 'wb') as f:
        with open(target_dir + 'MNIST_train_target.bin', 'wb') as g:
            for batch_idx, (data, target) in enumerate(train_loader):
                for by in data.storage().tolist():
                    f.write(struct.pack("@f", by))
                for by in target.storage().tolist():
                    g.write(struct.pack("@i", int(by)))
                if batch_idx % 6000 == 0:
                    print('[{}/{} ({:.0f}%)]'.format(
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader)))

import os
target_dir = 'data/bin/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    train()


