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
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    startTime = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       # transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    # skip tests
    #test_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                       transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,), (0.3081,))
    #                   ])),
    #    batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
    #self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias = False)
    fc1 = nn.Linear(784, 50)
    fc2 = nn.Linear(50, 10)
    optimizer = optim.SGD(None, lr=args.lr, momentum=args.momentum)

    def forward(x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 784)
        #x = self.fc1(x)
        x = F.relu(fc1(x))
        #x = F.dropout(x, training=self.training)
        x = fc2(x)
        return F.log_softmax(x, dim=1)

    def train(epoch):
        # model.train()
        tloss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
            data1 = Variable(data)
            target1 = Variable(target)
            optimizer.zero_grad()
            output = forward(data1)
            loss = F.nll_loss(output, target1)
            tloss = tloss + loss.data[0]
            loss.backward()
            optimizer.step()
        #    if ((batch_idx + 1) * len(data)) % args.log_interval == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), tloss / (batch_idx)))
        return tloss / len(train_loader)

    asdf = train(10)

print(run.src)

@stageTensor
def runX(x):
    return run(x)

print(runX.code)