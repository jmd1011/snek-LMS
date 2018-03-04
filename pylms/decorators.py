import sys
import ast as py_ast
import types
import parser
import inspect

from .py_to_sexpr import AstVisitor

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
        def __call__(self,*args):
            return func(*args)
    return Snippet()

def lms(func):
    class Snippet(object):
        def __init__(self):
            self.original = func
            #self.code = "foobar"
        def __call__(self,*args):
            return func(*args)
    return Snippet()

