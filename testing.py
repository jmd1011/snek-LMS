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

  init = newTensor(3)
  print(init)
  z = Variable(init)
  y = z + z
  out = F.nll_loss(y, 1)
  loss = out.backward()
  print(z)
  print(loss.data[0])

print(test.src)

@stageTensor
def testX(x):
  return test(x)

print(testX.code)

print(testX.Ccode)

testX(1)
