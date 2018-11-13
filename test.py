def runWhile (in1):
  x7 = None
  x8 = x7 = 3
  x9 = x7
  x10 = x9 < in1
  def genf0 ():
    x11 = x7
    x12 = x11 < in1
    return x12
  def genf1 ():
    x11 = x7
    x12 = x11 + 1
    x13 = x7 = x12
    None
  while genf0():
    genf1()

  x12 = x7
  return x12

print(runWhile(2))

def runLift (in1):
  x7 = None
  x8 = x7 = in1
  x9 = x7
  x10 = x9 > 0
  if x10:
    x11 = x7
    x12 = x11 + 1
    x13 = x7 = x12
    None
  else:
    x11 = x7
    x12 = x11 - 1
    x13 = x7 = x12
    None

  x12 = x7
  return x12

print(runLift(2))


def runX (in1):
  x0 = None
  x1 = in1 * 1
  x2 = in1 * x1
  x3 = in1 * x2
  x4 = x0 = x3
  x5 = x0
  return x5

print(runX(5))