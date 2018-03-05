import pytest

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
    assert(str(reify(lambda: power1(Rep("in"),3))) == 
        "[['val', x0, ['*', in, 1]], ['val', x1, ['*', in, x0]], ['val', x2, ['*', in, x1]], x2]")
    assert(str(reify(lambda: power2(Rep("in"),3))) == 
        "[['val', x0, ['*', in, 1]], ['val', x1, ['*', in, x0]], ['val', x2, ['*', in, x1]], x2]")

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

@lms
def foobar1(x):
    if (x == 0):
        print('yes')
    else:
        print('no')
    return x

def test_foobar1():
    assert(foobar1(7) == 7)

#@pytest.mark.skip(reason="not virtualizing print yet. also requires side effects")
def test_foobar1_staged():
    assert(str(reify(lambda: foobar1(Rep("in")))) == 
"""
[['val', x0, ['==', in, 0]], 
 ['val', x1, ['if', x0, [None], [None]]], in]
""".replace('\n','').replace('  ',' '))
#        "['if', ['==', in, 0], ['print' 'yes'], ['print' 'no']]")

def test_foobar1_rewrite():
    assert(foobar1.src == """

def foobar1(x):
    try:

        def then$1():
            print('yes')

        def else$1():
            print('no')
        __if((x == 0), then$1, else$1)
        __return(x)
    except NonLocalReturnValue as r:
        return r.value
""")

@lms
def foobar2(x):
    if x == 0:
        return "yes"
    else:
        return "no"

def test_foobar2():
    assert(foobar1(7) == 7)

def test_foobar2_staged():
    assert(str(reify(lambda: foobar2(Rep("in")))) == 
        "[['val', x0, ['==', in, 0]], ['val', x1, ['if', x0, 'yes', 'no']], x1]")

def test_foobar2_rewrite():
    assert(foobar2.src == """

def foobar2(x):
    try:

        def then$1():
            __return('yes')

        def else$1():
            __return('no')
        __if((x == 0), then$1, else$1)
    except NonLocalReturnValue as r:
        return r.value
""")


@lms
def loop1(n):
    x = 0
    while x < n: 
        x = x + 1
    return x

def test_loop1():
    assert(loop1(7) == 7)

# NOTE: this is still losing side effects (expected!)
def test_loop1_staged():
    assert(str(reify(lambda: loop1(Rep("in")))) == 
"""
[['val', x5, ['new']], 
 ['val', x6, ['set', x5, 0]], 
 ['val', x7, ['while', 
    [['val', x7, ['get', x5]], 
     ['val', x8, ['<', x7, in]], 
     x8], 
    [['val', x7, ['get', x5]], 
     ['val', x8, ['+', x7, 1]], 
     ['val', x9, ['set', x5, x8]], 
     None]]], 
 ['val', x8, ['get', x5]], x8]
""".replace('\n','').replace('  ',' ').replace('  ',' ').replace('  ',' '))

def test_loop1_rewrite(): ## FIXME: need to lift (selected?) variables
    assert(loop1.src == """

def loop1(n):
    try:
        x = __var()
        __assign(x, 0)

        def cond$1():
            return (__read(x) < n)

        def body$1():
            __assign(x, (__read(x) + 1))
        __while(cond$1, body$1)
        __return(__read(x))
    except NonLocalReturnValue as r:
        return r.value
""")        

