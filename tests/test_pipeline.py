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

@lms
def foobar(x):
    if (x == 0):
        print('yes')
    else:
        print('no')
    return x

@stage
def foobar1(x):
    return foobar(x)

def testFoobar1():
    assert(foobar1(3) == 3)

@lms
def loop(n):
    x = 0
    while x < n:
        x = x + 1
    return x

@stage
def loop1(n):
    return loop(n)

def testLoop1():
    assert(loop1(5) == 5)
