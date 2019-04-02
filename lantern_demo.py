from __future__ import print_function
from pylms import stage, lms

@lms
def run(dummy):
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import time

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--activateFunc', type=int, default=1, metavar='N',
                        help='1 = relu, else tanh')
    args = parser.parse_args()

    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False, **kwargs)

    fc1 = nn.Linear(784, 50)
    fc2 = nn.Linear(50, 10)
    optimizer = optim.SGD([fc1, fc2], lr=0.0005, momentum=0.0)

    @rep_fun
    def lossFun(x, target):
        x1 = x.view(-1, 784)
        x2 = F.relu(fc1(x1))
        x3 = fc2(x2)
        x4 = F.log_softmax(x3, dim=1)
        x5 = F.nll_loss(x4, target)
        return x5

    def train(epoch):
        tloss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data1 = Variable(data, volatile=True)
            target1 = Variable(target)
            optimizer.zero_grad()
            res = lossFun(data1, target1)
            res.backward()
            tloss = tloss + res.item()
            optimizer.step()
            tmp = tloss
            if (batch_idx + 1) % 6000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(train_loader),
                    100. * batch_idx / len(train_loader), tmp))
        return tloss / len(train_loader)

    idx = 0
    print("Start Training")
    while idx < 5:
        idx = idx + 1
        print('Epoch {}'.format(idx))
        train(idx)

print("==============================================================")
print("=======================ORIGINAL SOURCE========================")
print("==============================================================")
print(run.original_src)

print("==============================================================")
print("========================STAGED SOURCE=========================")
print("==============================================================")
print(run.src)

@stage
def runX(x):
    return run(x)

print("==============================================================")
print("===========================IR CODE============================")
print("==============================================================")
print(runX.code)

print("==============================================================")
print("========================GENERATED CODE========================")
print("==============================================================")
print(runX.Ccode)

print("==============================================================")
print("========================EXECUTING CODE========================")
print("==============================================================")
import time
start = time.time()
runX('')
stop = time.time()
print(int(stop - start))

print("==============================================================")
print("========================EXITING PROGRAM=======================")
print("==============================================================")
