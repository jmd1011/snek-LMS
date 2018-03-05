import sys
import ast as py_ast
import types
import parser
import inspect
import astunparse

from py4j.java_gateway import JavaGateway

from .py_to_sexpr import AstVisitor
from .lms_tree_rewriter import StagingRewriter

from .rep import *

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
            self.Ccode = self.gateway.jvm.sneklms.Main.compileMain(self.code)

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

