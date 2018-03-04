from py4j.java_gateway import JavaGateway

class SexpToC():

  def __init__(self):
    self.gateway = JavaGateway()

  def compile(self, code):
    ccode = self.gateway.jvm.sneklms.Main.compile(code)

    print(ccode)

compiler = SexpToC()
compiler.compile("1")
