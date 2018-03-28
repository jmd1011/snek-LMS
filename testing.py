from pylms import lms, stage, stageTensor
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

  z = torch.Tensor(2, 3)
  y = z + z
  y.print()

print(test.src)

@stageTensor
def testX(x):
  return test(x)

print(testX.code)
