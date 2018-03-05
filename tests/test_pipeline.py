import pytest

from pylms import lms, lmscompile, stage
from pylms.rep import *

# @stage
# def power1(b, x):
#     if (x == 0): return 1
#     else: return b * power1(b, x-1)

# def testPower():
#   assert(power1(3, 3) == "")