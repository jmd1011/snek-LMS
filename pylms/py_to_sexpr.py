import sys
import ast
import types
import parser
import inspect
import builtins
import os

class AstVisitor(ast.NodeVisitor):
    def __init__(self):
        self.f = open("test.out", "w")
#        super()

    def _print(self, s):
        self.f.write(s)

    def result(self):
    	self.f.flush()
    	f = open("test.out","r")
    	res = f.read()
    	f.close()
    	return res

    def visit_If(self, node):
        self._print("\n(if (")
        self.visit(node.test)
        self._print(")")
        flag = 0
        if(len(node.body) > 1):
            self._print("(")
            flag = 1
        y = node.body.pop()
        while(1):
            self.visit(y)
            if( len(node.body) == 0):
                break
            y = node.body.pop()
        if flag == 1:
            self._print(")")
            flag = 0
        if ( len(node.orelse) > 1):
            self._print("(")
            flag = 1
        y = node.orelse.pop()
        while(1):
            self.visit(y)
            if(len(node.orelse) ==0):
                break
            y = node.orelse.pop()
        if flag == 1:
            self._print(")")
        self._print(")\n")

    def visit_While(self, node):
        self._print("\n(while(")
        self.visit(node.test)
        self._print(")(")
        y = node.body.pop()
        while(1):
            self.visit(y)
            if ( len(node.body) == 0):
                break
            y = node.body.pop()
        self._print(")\n")

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        self._print("(def {0} (".format(node.name))
        for y in node.args.args:
            self._print(" {0} ".format(y.arg))
        self._print(") (\n")
            #visit all child nodes using self.generic_visit(child nodes...)
        self.generic_visit(node)
        self._print("\n))")

    def visit_Print(self, node):
        self._print("(print:")
        self.generic_visit(node)
        self._print(")")

    def visit_Str(self,node):
       self._print('"{0}"'.format(node.s))

    def visit_Arg(self, node):
       self._print("def {0} ".format(node.arg))

    def visit_Return(self, node):
        self._print("(return ")
        self.generic_visit(node)
        self._print(')')

    def visit_Num(self,node):
        z = str(node.n)
        self._print(' {0}'.format(z))

    def visit_BinOp(self, node):
        self._print("(")
        self.visit(node.op)
        self.visit(node.left)
        self.visit(node.right)
        self._print(')')

    def visit_Mult(self, node):
        self._print(" * ")

    def visit_Sub(self, node):
       self._print("- ")

    def visit_Add(self, node):
       self._print("+ ")

    def visit_Eq(self,node):
       self._print("== ")

    def visit_Lt(self, node):
       self._print("< ")

    def visit_Gt(self,node):
       self._print("> ")

    def visit_Lte(self,node):
       self._print("<= ")

    def visit_Gte(self,node):
       self._print(">= ")

    def visit_NotEq(self, node):
       self._print("!= ")

    def visit_Assign(self,node):
        self._print('(= ')
        self.generic_visit(node)
        self._print(")")

    def visit_Name(self, node):
        self._print("{0} ".format(node.id))
        self.generic_visit(node)

    def visit_Compare(self, node):
        y = node.ops.pop()
        self.visit(y)
        self.generic_visit(node)

    def visit_Attribute(self,node):
       self.visit(node.value)
       self._print("{0}".format(node.attr))

    def visit_Call(self, node):
        # print(ast.dump(node))
        self._print('(')
        if not isinstance(node.func, ast.Name):
            self._print('call ')
        self.generic_visit(node)
        self._print(')')

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
        # print(ast.dump(func_ast))
        AstVisitor().visit(func_ast)
        #for n in ast.walk(func_ast):
        #     # if isinstance(n, ast.If):
           #    print(ast.dump(n))
        # new_func_ast = AstVisitor().visit(func_ast)
        # ast.fix_missing_locations(new_func_ast)
        # exec(compile(new_func_ast, filename="<ast>", mode="exec"), globals())
        # return eval(func.__name__)
    elif isinstance(obj, types.MethodType):
        return NotImplemented
    else: return NotImplemented

######################################

# @sexp
# def power(b, x):
#     if (x == 0): return 1
#     else: return b * power(b, x - 1)


# @sexp
# def test(x):
#     model.eval()
#     if x == 0:
#         print("Hello")
#     else:
#         print("world!")
#     return x

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
