from pylms import lms, stage
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

  z = newTensor(2, 3)
  y = z + z
  y.print()

print(test.src)

@stage
def testX(x):
  return test(x)

print(testX.code)