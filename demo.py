from pylms import *
from pylms.rep import *
from pylms.nn_staging import *

@lms
def lifting_param(x):
  if x > 0:
    return x + 1
  else:
    return x - 1

print("======= Original code =======")
print(lifting_param.original_src)
print("======= Converted code ========")
print(lifting_param.src)
val = lifting_param(2)
assert(val == 3)
print("\n")

@stage
def runLift(x):
  return lifting_param(x)

print("======= SExpr ========")
print(runLift.code)
print("\n")
print("======= C/C++ code ========")
print(runLift.Ccode)
val = runLift(2)
assert(val == 3)

@lms
def testWhile(x):
  z = 3
  while z < x:
    z = z + 1
  return z

print("======= Original code =======")
print(testWhile.original_src)
print("======= Converted code ========")
print(testWhile.src)
val = testWhile(2)
assert(val == 3)
print("\n")

@stage
def runWhile(x):
  return testWhile(x)

print("======= SExpr ========")
print(runWhile.code)
print("\n")
print("======= C/C++ code ========")
print(runWhile.Ccode)
val = runWhile(10)
assert(val == 10)

@lms
def run(x):
  def power(n, k):
    if k == 0:
      return 1
    else:
      return n * \
        power(n, k - 1)
  res = power(x, 3)
  return res

print("======= Original code =======")
print(run.original_src)
print("======= Converted code ========")
print(run.src)
print("\n")

@stage
def runX(x):
  return run(x)

print("======= SExpr ========")
print(runX.code)
print("\n")
print("======= C/C++ code ========")
print(runX.Ccode)
val = runX(5)
assert(val == 125)
