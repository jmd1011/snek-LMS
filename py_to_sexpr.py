#/usr/bin/python3

import sys
import ast
import types
import parser
import inspect
import builtins
import virtualized
import os

class AstVisitor(ast.NodeVisitor):
    def _print(self, s):
        self.f.write(s)

    def __init__(self):
        self.f = open("test.out", "w")
        super()

    def visit_If(self, node):
        self.generic_visit(node)

    def visit_While(self, node):
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._print("(def {0}".format(node.name))

        #visit all child nodes using self.generic_visit(child nodes...)

        self._print(")")

        # Drop the decorator
        self.generic_visit(node)
        node.decorator_list = []

    def visit_Return(self, node):
        self.generic_visit(node)

    def visit_Name(self, node):
        self.generic_visit(node)

def sexp(obj):
    """
    Rep transforms the AST to annotated AST with Rep(s).
    TODO: What about Rep values defined inside of a function, rather than as an argument?
    TODO: How to lift constant value?
    TODO: How to handle `return`
    TODO: Handle sequence and side effects
    TODO: Assuming that there is no free variables in the function
    """
    if isinstance(obj, types.FunctionType):
        func = obj
        func_ast = ast.parse(inspect.getsource(func))
        AstVisitor().visit(func_ast)
        # for n in ast.walk(func_ast):
        #     # if isinstance(n, ast.If):
        #     print(ast.dump(n))
        # new_func_ast = AstVisitor().visit(func_ast)
        # ast.fix_missing_locations(new_func_ast)
        # exec(compile(new_func_ast, filename="<ast>", mode="exec"), globals())
        # return eval(func.__name__)
    elif isinstance(obj, types.MethodType):
        return NotImplemented
    else: return NotImplemented

######################################

@sexp
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

"""

@sexp
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x - 1)

==========

(def power (b x) (
    (if (== x 0) (return 1) (return (* b (call power b (- x 1))))
))
"""