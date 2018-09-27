from pylms import lms, lmscompile, stage
from pylms.rep import *
from pylms.nn_staging import *
# import numpy

@lms
def run(z):
	def power(x, n):
	    if n == 0:
	        return 1
	    else:
	        return x * power(x, n - 1)
	return power(2, z)

@stage
def runX(x):
    return run(x)

# @lms
# def loop(x):
#     y = 0
#     while y < x:
#         print(y)
#         y = y + 1
#     # continue
#     return y

# @stage
# def loopX(x):
#     return loop(x)

# @lms
# def test_for(train_loader):
#     x = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = Variable(data)
#         target = Variable(target)
#         new_data = data * data
#         x = x + 1
#     return x

# @stage
# def test_for2(x):
#     return test_for(x)

# @lms
# def lib(x):
#     optim.SGD(1, 2, 3)
#     return x

# @stage
# def libX(x):
#     return lib(x)

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
