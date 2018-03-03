#/usr/bin/python3

import sys
import ast
import types
import parser
import inspect
import builtins


var_counter = 0

def freshName():
    global var_counter
    n_var = var_counter
    var_counter += 1
    return "1_$vb" + str(n_var)

class ConditionRepChecker(ast.NodeVisitor):
    def __init__(self, reps):
        self.reps = reps
        self.hasRep = False
        super()

    def visit_Name(self, node):
        print("-------Name-------")
        if node.id == "RepString" or node.id == "RepInt":
            self.hasRep = True

def vIf(test, body, orelse, reps):
    print("----------IF----------")
    if(isinstance(test, bool)):
        print("No rep")
        if(test):
            # print(ast.dump(body()))
            print("True")
            return body()
        else:
            # print(type(orelse))
            # print(ast.dump(orelse()))
            print("False")
            # print(orelse())
            return orelse()
            # return orelse()
    else:
        print("Rep")
    print("--------End IF--------")

def parameterized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

def getClass(cls_name, mod_name = __name__):
    """
    Get Class definition from module, current module by default
    """
    modl = sys.modules[mod_name]
    Cls = getattr(modl, cls_name)
    return Cls

def getFunRepAnno(f):
    if isinstance(f, types.FunctionType):
        mod_ast = ast.parse(inspect.getsource(f))
        assert(isinstance(mod_ast, ast.Module))
        funcdef_ast = mod_ast.body[0] #TODO: always get the first one?
        assert(isinstance(funcdef_ast, ast.FunctionDef))
        return getFunRepAnno(funcdef_ast)

    if isinstance(f, ast.FunctionDef):
        ann_args = list(filter(lambda a: a.annotation, f.args.args))
        return dict(list(map(lambda a: (a.arg, a.annotation.id), ann_args)))

    raise NotImplementedError(f.__name__)

################################################

class IR(object): pass

class IRDef(IR):
    def __init__(self, name, args, body):
        self.name = name
        self.args = args
        self.body = body

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

class IRIntEq(IR):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class IRIf(IR):
    def __init__(self, cnd, thn, els):
        raise NotImplementedError("IRIf")

class IRWhile(IR):
    def __init__(self, cnd, body):
        self.cnd = cnd
        self.body = body

class IRRet(IR):
    def __init__(self, val):
        raise NotImplementedError("IRRet")

__if = IRIf
__while = IRWhile
__return = IRRet

################################################

class CodeGen(object): pass

class PyGenIRDef(object):
    def gen(self, irdef):
        bodycode = PyCodeGen(irdef.body).gen()
        return "def {0}({1}): return {2}".format(irdef.name, ",".join(irdef.args), bodycode)

class PyGenIRConst(object):
    def gen(self, irconst): return str(irconst.v)

class PyGenIRInt(object):
    def gen(self, irint): return str(irint.n)

class PyGenIRIntAdd(object):
    def gen(self, iradd):
        lhscode = PyCodeGen(iradd.lhs).gen()
        rhscode = PyCodeGen(iradd.rhs).gen()
        return "{0} + {1}".format(lhscode, rhscode)

class PyGenIRIntMul(object):
    def gen(self, irmul):
        print(irmul.lhs)
        print(irmul.rhs)
        lhscode = PyCodeGen(irmul.lhs).gen()
        rhscode = PyCodeGen(irmul.rhs).gen()
        return "{0} * {1}".format(lhscode, rhscode)

class PyGenIRIntEq(object):
    def gen(self, ireq):
        lhscode = PyCodeGen(ireq.lhs).gen()
        rhscode = PyCodeGen(ireq.rhs).gen()
        return "{0} == {1}".format(lhscode, rhscode)

class PyCodeGen(CodeGen):
    def __init__(self, ir):
        self.ir = ir

    def gen(self):
        clsName = "PyGen{0}".format(type(self.ir).__name__)
        Cls = getClass(clsName)
        return Cls().gen(self.ir)

    def ast(self):
        return ast.parse(self.gen())
    
    def dumpast(self):
        return ast.dump(self.ast())

class CCodeGen(CodeGen): pass

class SchemeGenIRDef(object):
    def gen(self, irdef):
        bodycode = SchemeCodeGen(irdef.body).gen()
        return "(define ({0} {1}) {2})".format(irdef.name, " ".join(irdef.args), bodycode)

class SchemeGenIRConst(object):
    def gen(self, irconst): return str(irconst.v)

class SchemeGenIRInt(object):
    def gen(self, irint): return str(irint.n)

class SchemeGenIRIntAdd(object):
    def gen(self, iradd):
        lhscode = SchemeCodeGen(iradd.lhs).gen()
        rhscode = SchemeCodeGen(iradd.rhs).gen()
        return "(+ {0} {1})".format(lhscode, rhscode)

class SchemeGenIRIntMul(object):
    def gen(self, irmul):
        lhscode = SchemeCodeGen(irmul.lhs).gen()
        rhscode = SchemeCodeGen(irmul.rhs).gen()
        return "(* {0} {1})".format(lhscode, rhscode)

class SchemeCodeGen(CodeGen):
    def __init__(self, ir):
        self.ir = ir

    def gen(self):
        clsName = "SchemeGen{0}".format(type(self.ir).__name__)
        Cls = getClass(clsName)
        return Cls().gen(self.ir)

################################################

class RepTyp(object): pass

#class RepInt(RepTyp, metaclass=RepTyp):
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
    def __eq__(self, m):
        if isinstance(m, RepTyp): m = m.__IR__()
        return IRIntEq(self.__IR__(), m)
class RepStr(RepTyp): pass

################################################

class StagingRewriter(ast.NodeTransformer):
    """
    StagingRewriter does two things:
    1) lift next-stage variables to be Rep
    2) virtualize primitives such as `if`, `while`, `for` and etc
    Note: Probably we may also rewrite operators such as `+` rather than overloading them.
    """
    def __init__(self, reps = {}):
        self.reps = reps
        super()
    def visit_If(self, node):
        # TODO: Virtualization of `if`
        # If the condition part relies on a staged value, then it should be virtualized.
        # print(ast.dump(node))

        # if node is of the form (test, body, orelse)

        # iter_fields lets me iterate through the contents of the if node
        # gives the child as a tuple of the form (child-type, object)
        # cond_node = node.test;
        # check for BoolOp and then Compare

        self.generic_visit(node)
        
        # vIf(node.test, node.body, node.orelse, self.reps)
        tBranch_name = freshName()
        eBranch_name = freshName()
        tBranch = ast.FunctionDef(name=tBranch_name, 
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.body, 
                                  decorator_list=[])
        eBranch = ast.FunctionDef(name=eBranch_name, 
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.orelse, 
                                  decorator_list=[])
        ast.fix_missing_locations(tBranch)
        ast.fix_missing_locations(eBranch)

        # self.vIf(node.test, tBranch, eBranch)
        # return node # COMMENT WHEN __if IS DONE

        # UNCOMMENT WHEN __if IS DONE
        # print(node.lineno)
        new_node = ast.Expr(ast.Call(
            func=ast.Name(id='vIf', ctx=ast.Load()), 
            args=[node.test, 
                  ast.Name(id=tBranch_name, ctx=ast.Load()), 
                  ast.Name(id=eBranch_name, ctx=ast.Load()),
                  ast.Dict(list(map(ast.Str, self.reps.keys())), 
                           list(map(ast.Str, self.reps.values()))),
                 ], 
            keywords=[]
        ))
        ast.fix_missing_locations(new_node)
        mod = [tBranch, eBranch, new_node]
        return mod
        #return ast.copy_location(mod, node)

    def visit_While(self, node):
        # TODO: Virtualization of `while`

        self.generic_visit(node)
        return node # COMMENT WHEN __while IS DONE

        # UNCOMMENT WHEN __while IS DONE
        # return ast.copy_location(ast.Call(func=ast.Name('__while', ast.Load()),
        #                                 args=[node.test, node.body],
        #                                 keywords=[]),
        #                         node)

    def visit_FunctionDef(self, node):
        # Retrieve the Rep annotations
        # TODO: handle multiple functions using a dict
        self.reps = getFunRepAnno(node)
        self.generic_visit(node)
        node.decorator_list = [] # Drop the decorator
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        # TODO: just a poor hack to make power work
        if ast.dump(node.value) == ast.dump(ast.Num(1)):
            ret = ast.copy_location(ast.Return(value=ast.Call(func=ast.Name(id='RepInt', ctx=ast.Load()),
                                                               args=[ast.Num(1)],
                                                               keywords=[])),
                                     node)

            return ret
        return node # COMMENT WHEN __return IS DONE

        # UNCOMMENT WHEN __return IS DONE
        # return ast.copy_location(ast.Call(func=ast.Name('__return', ast.Load()),
        #                                 args=[node.value],
        #                                 keywords=[]),
        #                         node)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id in self.reps:
            return ast.copy_location(ast.Call(func=ast.Name(id=self.reps[node.id], ctx=ast.Load()),
                                              args=[ast.Str(s=node.id)],
                                              keywords=[]),
                                    node)
        return node

def lms(obj, *args, **kwargs):
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
        mod_ast = ast.parse(inspect.getsource(func))

        # GW: do you want to compare unfix/fix version of new_mod_ast?
        #     rather than mod_ast vs new_mod_ast
        print("before fixing, ast looks like this:\n\n{0}".format(ast.dump(mod_ast)))
        print("========================================================")

        new_mod_ast = StagingRewriter().visit(mod_ast)
        ast.fix_missing_locations(new_mod_ast)

        print("after fixing, ast looks like this:\n\n{0}\n\n".format(ast.dump(new_mod_ast)))

        exec(compile(new_mod_ast, filename="<ast>", mode="exec"), globals())
        return eval(func.__name__)
    elif isinstance(obj, types.MethodType):
        return NotImplemented
    else: return NotImplemented

def __return(valueNode):
    # idea: we evaluate whatever is being returned down to a single statement (e.g., b * b * ...)
    # note that because this is wrapped inside this __return function, it *should* (might)
    # handle control flow issues because only the first return is transformed
    ret = exec(compile(valueNode, filename="<ast>", mode="exec"), globals())
    return ret

@parameterized
def Specalize(f, Codegen, *args, **kwargs):
    """
    Specalize transforms the annotated IR to target language.
    Note: f must be a named function
    TODO: f's argument names may different with variables in the inner body
    """
    fun_name = f.__name__
    fun_args = inspect.getfullargspec(f).args
    rep_anno = getFunRepAnno(f)
    rep_args = [getClass(rep_anno[farg])(farg) for farg in fun_args]
    irbody = f(*rep_args)
    codegen = Codegen(IRDef(fun_name, fun_args, irbody))
    codestr = codegen.gen()
    exec(codestr, globals())
    return eval(fun_name)

################################################

def giveBack(x):
    return x

"""
TODO: User can provide return type.
TODO: User can even provide return type as union;
      eg, power(b: RepInt, x) -> RepInt | RepStr
"""
@lms
def power(b : RepInt, x) -> RepInt:
    if (x == 0): return 1
    else: return b * power(b, x-1)

"""
@lms
def power(b : RepInt, x):
    __if(x == 0, __return([1]), __return(b * power(b, x - 1)))

def __if(cond, body, orelse):
    #look at cond
    if (cond is staged): #if cond has an operand with Rep[T]
        generate if () ...
    else:
        if (cond):
            ret = visit(body) #generate code for then branch (maybe)
            if isinstance(ret, ast.Return):
                return eval(ret)
            else:
                return ret
        else:
            visit(orelse)

"""

"""
Explaination: intuitively, the instrumented AST of `power` looks like `stagedPower`.
`if` is a current-stage expression, so it is not virtualized, and will be executed as normal code.
But `b` is a next-stage variable, so it became `RepInt(b)`. `1` also lifted to `RepInt(1)`
because we (implicitly) require that the types of both branches should same.
`*` __mul__ operator should be overloaded by RepInt.
By running `stagedPower`, we obtain the IR that will be used later by code generator.
"""
def stagedPower(b, x):
    if (x == 0): return RepInt(1)
    else: return b * stagedPower(b, x - 1)

################################################

"""
Ideally, user could specify different code generators for different targer languages.
The code generator translates IR to string representation of target language.
"""
@Specalize(PyCodeGen)
def snippet(b: RepInt):
    return power(b, 3)

assert(snippet(3) == 27) # Here we can just use snippet

"""
Explaination: decorating `snippet` with @Specalize is equivalent to:
"""
def snippet2(b : RepInt): return stagedPower(b, 3)
power3 = Specalize(PyCodeGen)(snippet2)
assert(power3(3) == 27)

