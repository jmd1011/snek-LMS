from pylms import lms, lmscompile, stage
from pylms.rep import *
from pylms.nn_staging import *
# import numpy


# @lms
# def power(x, n):
#     if n == 0:
#         return 1
#     else:
#         return x * power(x, n - 1)

# @stage
# def power3(x):
#     return power(x, 3)

@lms
def run(x):
  def mul(a, b):
    return a * b
  return __call_staged(mul, x, x)

@stage
def runX(x):
    return run(x)



print("======= Power original code =======")
print(run.original_src)
print("======= Power converted code ========")
print(run.src)
print("\n")
print("======= Power3 IR ========")
print(runX.code)
print("\n")
print("======= Power3 C/C++ code ========")
print(runX.Ccode)

# print(power3(5))

# print("======= Loop converted code ========")
# print(loop.src)

# print("======= LoopX IR ========")
# print(loopX.code)

# print("running loop(5)")
# print(loop(5))

# print("running loopX(5)")
# print(loopX(5))

# print("======= lib converted code ========")
# print(lib.src)

# print("======= libX IR ========")
# print(libX.code)

# print("running lib(0)")
# print(lib(0))

# print("running libX(0)")
# print(libX(0))

# print("running loopX(5)")
# print(loopX(5))

# print("======= Floop converted code ========")
# print(floop.src)

# print("======= FloopX IR ========")
# print(floopX.code)

# print("running floop(5)")
# print(floop(5))

# print("running floopX(5)")
# print(floopX(5))
