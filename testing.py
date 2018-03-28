from pylms import lms, stage, stageTensor, ast
from pylms.rep import Rep

@lms
def test(x):
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torchvision import datasets, transforms
  from torch.autograd import Variable
  import time

  z = Variable(newTensor(2, 3))
  y = z + z
  return print(y)

print(test.src)

@stageTensor
def testX(x):
  return test(x)

print(testX.code)

print(testX.Ccode)
