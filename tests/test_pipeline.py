import pytest

from pylms import lms, lmscompile, stage
from pylms.rep import *

@stage
def power1(b, x):
    if (x == 0): return 1
    else: return b * power1(b, x-1)

def testPowerManual():
    import module_power1
    assert(module_power1.x1(3,3) == 27)

def testPower():
    assert(power1(3, 3) == 27)
