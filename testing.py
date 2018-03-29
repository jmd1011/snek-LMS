from pylms import lms, stage, stageTensor, ast
from pylms.rep import Rep

# @lms
# def t(x):
#   print("{}{}{}".format("0","1","2"))
#   # __printf("{}{}{}", ["0","1","2"])

# print(t.src)

# @stageTensor
# def tX(x):
#   return t(x)

# print(tX.code)

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
  # print("{}".format(z))

print(test.src)

@stageTensor
def testX(x):
  return test(x)

print(testX.code)

print(testX.Ccode)

testX(1)
