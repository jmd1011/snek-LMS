from pylms import lms, lmscompile, stage
from pylms.rep import *

@lms
def power(x, n):
    if n == 0:
        return 1
    else:
        return x * power(x, n - 1)

@stage
def power3(x):
    return power(x, 3)

print("======= Power converted code ========")
print(power.src)
print("\n")
print("======= Power3 IR ========")
print(power3.code)
print("\n")
print("======= Power3 C/C++ code ========")
print(power3.Ccode)
