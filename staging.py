#/usr/bin/python3

import sys
import ast
import types
import parser
import inspect

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

################################################

class IR: pass

class IRConst(IR):
    def __init__(self, v):
        self.v = v

class IRInt(IR):
    def __init__(self, n):
        self.n = n

class IRIntAdd(IR):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class IRIntMul(IR):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class IRIf:
    def __init__(self, cnd, thn, els): NotImplemented

class IRRet
    def __init__(self, val): NotImplemented

################################################

class PyGenIRConst:
    def gen(self, irconst): return str(irconst.v)

class PyGenIRInt:
    def gen(self, irint): return str(irint.n)

class PyGenIRIntAdd:
    def gen(self, iradd):
        lhscode = PyCodeGen(iradd.lhs).gen()
        rhscode = PyCodeGen(iradd.rhs).gen()
        return "{0} + {1}".format(lhscode, rhscode)

class PyGenIRIntMul:
    def gen(self, irmul):
        lhscode = PyCodeGen(irmul.lhs).gen()
        rhscode = PyCodeGen(irmul.rhs).gen()
        return "{0} * {1}".format(lhscode, rhscode)

class CodeGen: pass

class PyCodeGen(CodeGen):
    def __init__(self, ir):
        self.ir = ir
    def gen(self):
        clsName = "PyGen{0}".format(type(self.ir).__name__)
        modl = sys.modules[__name__]
        Cls = getattr(modl, clsName)
        return Cls().gen(self.ir)

class CCodeGen(CodeGen):
    raise NotImplementedError()

################################################

class RepTyp: pass

class RepInt(RepTyp):
    def __init__(self, n):
        self.n = n
    def __IR__(self):
        if isinstance(self.n, int): return IRConst(self.n)
        else: return IRInt(self.n)
    def __add__(self, m):
        if isinstance(m, RepTyp): m = m.__IR__()
        return IRIntAdd(self.__IR__(), m)
    def __mul__(self, m):
        if isinstance(m, RepTyp): m = m.__IR__()
        return IRIntMul(self.__IR__(), m)

################################################

class RepFunc():
    def __init__(self, func, *args, **kwargs):
        self.func = func
        #TODO

class CompiledCode:
    def __init__(self):
        pass

@parametrized
def Rep(obj, *args, **kwargs):
    """
    Rep transforms the AST to annotated AST with Rep(s).
    TODO: What about Rep values defined inside of a function, rather than as an argument?
    TODO: lift?
    TODO: How to handle `return`?
    TODO: sequence and side effects?
    """
    if isinstance(obj, types.FunctionType):
        return RepFunc(obj, *args, **kwargs)
    else: return NotImplemented

@parametrized
def Specalize(ast, Codegen, *args, **kwargs):
    """
    Specalize transforms the annotated IR to target language.
    """
    codegen = Codegen()
    def f(b): return eval(codegen.gen(ast))
    return f

################################################

"""
TODO: Does user need to provide Rep annotation on returned value?
"""
@Rep(b = RepInt)
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

"""
Intuitively, the instrumented AST of `power` looks like `stagedPower`.
`if` is a current-stage expression, so it is not virtualized, and will be executed as normal code.
But `b` is a next-stage variable, so it became `RepInt(b)`. `1` also lifted to `RepInt(1)`
because we (implicitly) require that the types of both branches should same.
`*` __mul__ operator should be overloaded by RepInt.
By running `stagedPower`, we obtain the IR that will be used later by code generator.
"""
def stagedPower(b, x):
    if (x == 0): return RepInt(1)
    else: return RepInt("b") * stagedPower(RepInt("b"), x - 1)

################################################

"""
Ideally, user could specify different code generators for different targer languages.
The code generator translates IR to string representation of target language.
"""
@Specalize(PyCodeGen, b = RepInt)
def snippet(b):
    return power(b, 3)

def specializedSnippet(b):
    return b * b * b * 1

ir = stagedPower(0, 3) # 0 is just a dummy value
print(PyCodeGen(ir).gen())
