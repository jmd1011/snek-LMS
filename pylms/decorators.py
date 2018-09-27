import sys
import ast as py_ast
import types
import parser
import inspect
import astunparse

from py4j.java_gateway import JavaGateway

from .py_to_sexpr import AstVisitor
from .lms_tree_rewriter import ScopeAnalysis, StagingRewriter

from .rep import *
from .nn_staging import *
from .onnx_staging import *

sys.path.insert(0, 'gen')

def ast(func):
    """
    Export a function AST to S-Expressions
    """
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.ast = py_ast.parse(inspect.getsource(func))
            print("ast:\n{}".format(py_ast.dump(self.ast)))
            visitor = AstVisitor()
            visitor.visit(self.ast)
            self.code = visitor.result().replace('\n','').replace('  ',' ').replace('( ','(').replace(' )',')').replace(')(',') (')
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.gen(self.code, "gen", self.moduleName)

        def __call__(self,*args):
            # return None
            return func(*args)

    return Snippet()

def lms(func):
    """
    LMS-like virtualization (if becomes __if() and so on)
    """
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.original_src = inspect.getsource(func)
            self.original_ast = py_ast.parse(self.original_src)
            scope = ScopeAnalysis()
            scope.visit(self.original_ast)
            visitor = StagingRewriter(scope)
            self.ast = visitor.visit(self.original_ast)
            py_ast.fix_missing_locations(self.ast)
            self.src = astunparse.unparse(self.ast)
            print("finished, src:\n{}".format(self.src))
            exec(compile(self.ast, filename="<ast>", mode="exec"), globals())
            self.func = eval(func.__name__)

            #self.code = "foobar"
        def __call__(self,*args):
            return self.func(*args)

    return Snippet()

def toSexpr(l):
    if not isinstance(l, list):
        return l
    if len(l) > 0 and isinstance(l[0], list):
        stm = l[0]
        if stm[0] == 'let':
            rhs = toSexpr(stm[2])
            body = toSexpr(l[1:])
            return ['let', stm[1], rhs, body]
        else:
            raise Exception()
    elif len(l) > 3 and l[0] == 'def':
        return ['def', l[1], l[2], toSexpr(l[3])]
    elif len(l) > 3 and l[0] == 'for_dataloader':
        return ['for_dataloader', l[1], l[2], toSexpr(l[3])]
    elif len(l) > 2 and l[0] == 'while':
        cond = toSexpr(l[1])
        body = toSexpr(l[2])
        return ['while', cond, body]
    elif len(l) > 1 and str(l[0]) == 'begin':
        return  ['begin'] + [toSexpr(l[1:])]
    elif len(l) > 3 and str(l[0]) == 'if':
        cond = toSexpr(l[1])
        then = toSexpr(l[2])
        oelse = toSexpr(l[3])
        return ['if', cond, then, oelse]
    elif len(l) == 1:
        return l[0]
    else:
        return l

def stage(func):
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.pcode = toSexpr(reify(lambda: func(Rep("in"))))
            self.code = "(def {} (in) (begin {}))".format(func.__name__, str(self.pcode).replace('[','(').replace(']',')').replace("'", '').replace(',', ''))
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.gen(self.code, "gen", self.moduleName)

        def __call__(self, *args): #TODO naming
            exec("import {} as foo".format(self.moduleName), globals())
            return foo.x1(*args)
            # return None

    return Snippet()


def stageTensor(func):
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.pcode = toSexpr(reify(lambda: func(Rep("in"))))
            self.code = "(def {} (in) (begin {}))".format(func.__name__, str(self.pcode).replace('[','(').replace(']',')').replace("'", '').replace(',', ''))
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.genT(self.code, "gen", self.moduleName)

        def __call__(self, *args): #TODO naming
            exec("import {} as foo".format(self.moduleName), globals())
            return foo.x1(*args)
            # return None

    return Snippet()

def lanternRun(func):
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.pcode = toSexpr(reify(lambda: func(Rep("in"))))
            self.code = "(def {} (in) (begin {}))".format(func.__name__, str(self.pcode).replace('[','(').replace(']',')').replace("'", '').replace(',', ''))
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.loadModel(self.code, "gen", self.moduleName)

        def __call__(self, *args): #TODO naming
            exec("import {} as foo".format(self.moduleName), globals())
            return foo.x1(*args)
            # return None

    return Snippet()

def lmscompile(func):
    """
    Compile LMS function
    """
    # if not isinstance(func, types.FunctionType):
    #     return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.code = str(reify(lambda: func(Rep("in"))))
            # obtain sexpr via .replace("[","(").replace("]",")")

        def __call__(self,*args):
            return self.func(*args)

    return Snippet()

