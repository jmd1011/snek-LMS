from pylms import lms
from pylms.rep import *

# power doesn't need virtualization
# sanity check that @lms doesn't mess up

def power1(b, x):
    if (x == 0): return 1
    else: return b * power1(b, x-1)

@lms
def power2(b, x):
    if (x == 0): return 1
    else: return b * power2(b, x-1)

def test_power():
    assert(power1(2,3) == 8)
    assert(power2(2,3) == 8)

def test_power_staged():
    assert(str(power1(Rep("in"),3)) == "['*', in, ['*', in, ['*', in, 1]]]")
    assert(str(power2(Rep("in"),3)) == "['*', in, ['*', in, ['*', in, 1]]]")
