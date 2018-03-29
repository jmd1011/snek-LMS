from pylms import lms, stage, stageTensor, ast
from pylms.rep import Rep

# @ast
# def t():
#   x = "hi"

@lms
def test(x):
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torchvision import datasets, transforms
  from torch.autograd import Variable
  import time

  # z = Variable(newTensor(2, 3))
  # y = z + z
  kwargs = {}
  train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       # transform=transforms.ToTensor()),
        batch_size=10, shuffle=False, **kwargs)

print(test.src)

@stageTensor
def testX(x):
  return test(x)

print(testX.code)

# print(testX.Ccode)
