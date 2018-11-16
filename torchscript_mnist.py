from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from pylms import *

def run():
    # outer_start = time.time()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       # transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    class Net(torch.jit.ScriptModule):
        __constants__ = ['activateFunc']
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 50)
            self.fc2 = nn.Linear(50, 10)
            self.activateFunc = args.activateFunc

        @torch.jit.script_method
        def forward(self, x):
            x1 = x.view([-1, 784])

            if self.activateFunc == 1:
                x2 = F.relu(self.fc1(x1))
                x3 = self.fc2(x2)
                x4 = F.log_softmax(x3, dim=1)
                res = x4
            else:
                x6 = F.tanh(self.fc1(x1))
                x7 = self.fc2(x6)
                x8 = F.log_softmax(x7, dim=1)
                res = x8
            return res

    model = Net()
    # if args.cuda:
    #     model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch):
        # model.train()
        tloss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            tloss += loss.data.item()
            loss.backward()
            optimizer.step()
            # if (batch_idx * len(data) + 1) % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data) + 1, len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), tloss / (batch_idx)))
        return tloss / (batch_idx)

    astart = time.time()
    for epoch in range(1, args.epochs + 1):
        # start = time.time()
        train(epoch)
        # stop = time.time()
        # print('Training completed in {} sec ({} sec/image)'.format(int(stop - start), (stop - start)/60000))
    astop = time.time()
    print('All training completed in {} sec'.format(int(astop - astart)))
    # print('Total time: {}'.format(int(astop - outer_start)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
    ## Note NEED default to be the same as total data length, or 1/10 of the total data length
    parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--activateFunc', type=int, default=1, metavar='N',
                    help='1 = relu, else tanh')
    args = parser.parse_args()
    run()
