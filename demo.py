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

@lms
def loop(x):
    y = 0
    while y < x:
        print(y)
        y = y + 1
    # continue
    return y

@stage
def loopX(x):
    return loop(x)


# @lms
# def floop(x):
#     y = 0
#     for i in x:
#         print(y)
#         y = y + 1
#     return y

# @stage
# def floopX(x):
#     return floop(x)


# print("======= Power converted code ========")
# print(power.src)
# print("\n")
# print("======= Power3 IR ========")
# print(power3.code)
# print("\n")
# print("======= Power3 C/C++ code ========")
# print(power3.Ccode)



print("======= Loop converted code ========")
print(loop.src)

print("======= LoopX IR ========")
print(loopX.code)

print("running loop(5)")
print(loop(5))

print("running loopX(5)")
print(loopX(5))


# print("======= Floop converted code ========")
# print(floop.src)

# print("======= FloopX IR ========")
# print(floopX.code)

# print("running floop(5)")
# print(floop(5))

# print("running floopX(5)")
# print(floopX(5))