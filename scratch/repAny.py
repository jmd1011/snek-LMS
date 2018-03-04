import parser
import inspect

counter = 0

def freshName():
  global counter
  counter += 1
  return "x{}".format(counter)

def buildspaces(tab):
  str = ""

  for i in range(tab):
    str += " "

  return str

class RepInt(object):
  def __init__(self, tab=0):
    self.val = freshName()
    print "{}int {};".format(buildspaces(tab), self.val)

  def add(self, y, tab=0):
    if type(y) is RepString:
      return y.add2(self, tab)

    res = RepInt(tab)
    print "{}{} = {} + {};".format(buildspaces(tab), res.val, self.val, y.val)
    return res

  def read(self, tab=0):
    # res = RepInt()
    print "{}fin >> {};".format(buildspaces(tab), self.val)
    return self

  def print1(self, tab=0):
    print "{}std::cout << {};".format(buildspaces(tab), self.val)
    return

class RepString(object):
  def __init__(self, tab=0):
    self.val = freshName()
    print "{}string {};".format(buildspaces(tab), self.val)

  def add(self, y, tab=0):
    res = RepString()
    print "{}{} = {} + {};".format(buildspaces(tab), res.val, self.val, y.val)
    return res

  def add2(self, y, tab=0):
    res = RepString()
    print "{}{} = {} + {};".format(buildspaces(tab), res.val, y.val, self.val)
    return res

  def read(self, tab=0):
    # res = RepString()
    print "{}fin >> {}".format(buildspaces(tab), res.val)
    return self

  def print1(self, tab=0):
    print "{}std::cout << {};".format(buildspaces(tab), self.val)
    return

def staged_read(tab=0):
  res = RepInt(tab)
  print "{}fin >> {};".format(buildspaces(tab), res.val)
  return res

def staged_read_str(tab=0):
  res = RepString(tab)
  print "{}fin >> {};".format(buildspaces(tab), res.val)
  return res