from pylms import *
from pylms.rep import *
from pylms.nn_staging import *

@lms
def run(x):
  @staged
  def foo(a, b):
    return a + b
  z = foo(1, x)
  return z

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
print(runX(2))