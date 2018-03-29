import pytest

from pylms import lms, lmscompile
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
    assert(lmscompile(lambda x: power1(x,3)).code ==
        "['begin', ['let', x0, ['*', in, 1]], ['let', x1, ['*', in, x0]], ['let', x2, ['*', in, x1]], x2]")
    assert(lmscompile(lambda x: power2(x,3)).code ==
        "['begin', ['let', x0, ['*', in, 1]], ['let', x1, ['*', in, x0]], ['let', x2, ['*', in, x1]], x2]")

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

# @pytest.mark.skip(reason="careful: print is now lifted!")
def test_foobar1():
   assert(lmscompile(lambda _: foobar1(7)).code == """['begin', ['let', x0, ['print', '"no"']], 7]""")

def test_foobar1_staged():
    assert(lmscompile(foobar1).code ==
"""
['begin', ['let', x0, ['==', in, 0]],
 ['let', x1, ['if', x0,
  ['begin', 'begin', ['let', x1, ['print', '"yes"']], None],
  ['begin', 'begin', ['let', x1, ['print', '"no"']], None]]], in]
""".replace('\n','').replace('  ',' ').replace('  ',' '))

#        "['if', ['==', in, 0], ['print' 'yes'], ['print' 'no']]")

def test_foobar1_rewrite():
    assert(foobar1.src == """

def foobar1(x):
    try:

        def then$1():
            __print('yes')

        def else$1():
            __print('no')
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
    assert(foobar2(7) == "no")

def test_foobar2_staged():
    assert(lmscompile(foobar2).code ==
        """['begin', ['let', x0, ['==', in, 0]], ['let', x1, ['if', x0, ['begin', 'begin', 'yes'], ['begin', 'begin', 'no']]], x1]""")

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

def test_loop1_staged():
    assert(lmscompile(loop1).code ==
"""
['begin', ['let', x5, ['new']],
 ['let', x6, ['set', x5, 0]],
 ['let', x7, ['while',
    ['begin', ['let', x7, ['get', x5]],
     ['let', x8, ['<', x7, in]],
     x8],
    ['begin', ['let', x7, ['get', x5]],
     ['let', x8, ['+', x7, 1]],
     ['let', x9, ['set', x5, x8]],
     None]]],
 ['let', x8, ['get', x5]], x8]
""".replace('\n','').replace('  ',' ').replace('  ',' ').replace('  ',' '))

def test_loop1_rewrite():
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

