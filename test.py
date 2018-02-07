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

class RepIntD(object):
  def __init__(self, f):
    # code = "1+1"
    # ast = parser.expr(code)
    # print "ast: {}".format(ast.tolist())
    # lines = inspect.getsource(f)
    # print "==============ORIGINAL=============="
    # print lines
    # print "==============CLEANED==============="
    # lines2 = "\n".join(map(lambda x: x[2:], filter(lambda y: "def" not in y, filter(lambda x: "@" not in x, lines.split("\n")))))
    # print lines2
    # # print "{}".format(filter(lambda x: "@" not in x, lines))
    # print "==============AST==============="
    # print parser.expr(lines2)
    return

  def __call__(self):
    return staged_read(2)

class RepStringD(object):
  def __init__(self, f):
    return

  def __call__(self):
    return staged_read_str(2)

def plus(x, y):
  x.add(y)

@RepIntD
def readInt():
  res = readingInt()

  if res > 0:
    res = 1
  else:
    res = 0

  return

@RepStringD
def readString():
  return

def readSchema():
  f = open("data.schema", "r")
  schema = f.readline().split(",")
  f.close()

  return schema

def startSnippet():
  print "#include<fstream>"
  print "#include<iostream>"
  print "#include<string>"
  print ""
  print "int main() {"
  print "  fstream fin;"
  print "  fin.open(\"data.csv\");"
  return
def endSnippet():
  print "  fin.close();"
  print "}"
  print ""

startSnippet()
schema = readSchema()
vals = map(lambda x: readInt() if x == "int" else readString(), schema)
for val in vals:
  val.print1(2)
# map(lambda x: x.print1(), vals)
endSnippet()
