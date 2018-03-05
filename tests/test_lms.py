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

def test_power_rewrite():
    assert(power2.src == """

def power2(b, x):
    try:

        def then$1():
            __return(1)

        def else$1():
            __return((b * power2(b, (x - 1))))
        __if((x == 0), then$1, else$1)
    except NonLocalReturnValue as r:
        return r.value
""")

# FIXE: this is wrong!

@lms
def foobar1(x):
    if (x == 0):
        print('yes')
    else:
        print('no')
    return x

def test_foobar1():
    assert(foobar1(7) == 7)

def test_foobar1_rewrite():
    assert(foobar1.src == """

def foobar1(x):
    try:

        def then$2():
            print('yes')

        def else$2():
            print('no')
        __if((x == 0), then$2, else$2)
        __return(x)
    except NonLocalReturnValue as r:
        return r.value
""")

