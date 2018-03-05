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
            visitor = AstVisitor()
            visitor.visit(self.ast)
            self.code = visitor.result().replace('\n','').replace('  ',' ').replace('( ','(').replace(' )',')').replace(')(',') (')
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.gen(self.code, "gen", self.moduleName)

        def __call__(self,*args):
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
            visitor = StagingRewriter()
            self.ast = visitor.visit(self.original_ast)
            py_ast.fix_missing_locations(self.ast)
            self.src = astunparse.unparse(self.ast)
            exec(compile(self.ast, filename="<ast>", mode="exec"), globals())
            self.func = eval(func.__name__)

            #self.code = "foobar"
        def __call__(self,*args):
            return self.func(*args)

    return Snippet()

def stage(func):
    if not isinstance(func, types.FunctionType):
        return NotImplemented

    class Snippet(object):
        def __init__(self):
            self.original = func
            self.original_src = inspect.getsource(func)
            self.original_ast = py_ast.parse(self.original_src)
            scope = ScopeAnalysis()
            scope.visit(self.original_ast)
            visitor = StagingRewriter()
            self.ast = visitor.visit(self.original_ast)
            py_ast.fix_missing_locations(self.ast)
            self.src = astunparse.unparse(self.ast)
            exec(compile(self.ast, filename="<ast>", mode="exec"), globals())
            self.func = eval(func.__name__)

            self.original = self.func
            self.ast = py_ast.parse(inspect.getsource(func))
            visitor = AstVisitor()
            visitor.visit(self.ast)
            self.code = visitor.result().replace('\n','').replace('  ',' ').replace('( ','(').replace(' )',')').replace(')(',') (')
            self.gateway = JavaGateway()
            self.moduleName = 'module_{}'.format(func.__name__)
            self.Ccode = self.gateway.jvm.sneklms.Main.gen(self.code, "gen", self.moduleName)

        def __call__(self, *args):
            exec("import {} as foo".format(self.moduleName), globals())
            return foo.x1(*args)

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

