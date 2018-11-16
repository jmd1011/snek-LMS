
@lms
def run():
    from __future__ import print_function
    from pylms import *
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import time
    import argparse
    # outer_start = time.time()

    # Training settings
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
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
        batch_size=1, shuffle=False, **kwargs)

    class Net(torch.jit.ScriptModule):
        # __constants__ = ['activateFunc']
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 50)
            self.fc2 = nn.Linear(50, 10)
            # self.activateFunc = args.activateFunc

        @torch.jit.script_method
        def forward(self, x):
            x1 = x.view([-1, 784])
            x2 = F.relu(self.fc1(x1))
            x3 = self.fc2(x2)
            x4 = F.log_softmax(x3, dim=1)
            return x4

    model = Net()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch):
        tloss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data1 = Variable(data)
            target1 = Variable(target)
            optimizer.zero_grad()
            output = model(data1)
            loss = F.nll_loss(output, target1)
            tloss += loss.data.item()
            loss.backward()
            optimizer.step()
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
    # print(run.src)
    # start = time.time()
    run()
    # end = time.time()
    # print('Total time: {}'.format(int(end - start)))
