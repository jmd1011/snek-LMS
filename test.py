def runWhile (in1):
  x7 = None
  x8 = x7 = 0
  x9 = x7
  x10 = x9 < in1
  def genf0 ():
    x11 = x7
    x12 = x11 < in1
    return x12
  while genf0():
    x11 = x7
    x12 = x11 + 1
    x13 = x7 = x12
    None


  x12 = x7
  return x12

print(runWhile(10))