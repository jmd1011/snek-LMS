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
from pylms.rep import *

@lms
def f(x):
  # optimizer = optim.SGD(None, lr=None, momentum=None)
  fc1 = nn.Linear(320, 50)
  return fc1(x)

@stage
def fX(x):
  return f(x)

print(f.src)
# print(reify(lambda: f(Rep("in"))))
print(fX.code)
