from __future__ import print_function
from pylms import lms, stage, stageTensor
from pylms.rep import Rep

@lms
def run(train_loader):
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import time
    from pylms import lms, stage
    from pylms.rep import Rep

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
                        help='how many batches to wait before logging training status')
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
    optimizer = optim.SGD([fc1.weight, fc1.bias, fc2.weight, fc2.bias], lr=args.lr, momentum=args.momentum)

    def forward(x):
        x1 = x.view(-1, 784)
        x2 = F.relu(fc1(x1))
        x3 = fc2(x2)
        return F.log_softmax(x3, dim=1)

    def train(epoch):
        tloss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data1 = Variable(data, volatile=True)
            target1 = Variable(target)
            optimizer.zero_grad()
            output = forward(data1)
            loss = F.nll_loss(output, target1)
            tloss = tloss + loss.data[0]
            loss.backward()
            optimizer.step()
            tmp = tloss
            if ((batch_idx + 1) * len(data)) % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), tmp / batch_idx))
        return tloss / len(train_loader)

    asdf = train(args.epochs)

print(run.src)

@stageTensor
def runX(x):
    return run(x)

print(runX.code)
