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

@ast
def test(lst):
  for l in lst:
    print(l)
  pass

@lms
def f(x):
  fc1 = nn.Linear(320, 50)
  return fc1(x)

print(f.src)
print(reify(lambda: f(Rep("in"))))
