import pytest

from pylms import lms, lmscompile, stage
from pylms.rep import *

@lms
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

@stage
def power3(b):
    return power(b, 3)

def testPower():
    assert(power3(3) == 27)

@stage
def foobar1(x):
    if (x == 0):
        print('yes')
    else:
        print('no')
    return x

def testFoobar1():
    assert(foobar1(3) == 3)
