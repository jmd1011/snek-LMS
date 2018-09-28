import inspect

__all__ = [
    'reflect', 'reflectTensor', 'reify', 'fresh', 'rep_tuple',
    'Rep', 'RepTensor', 'newTensor', 'freshTensor', 'RepTuple', 'reflectTuple', 'reflectDef',
    'NonLocalReturnValue', 'NonLocalBreak', 'NonLocalContinue',
    '__if', '__while', '__def_staged', '__call_staged', '__return', '__print', '__printf',
    '__var', '__assign', '__read', '__len',
    '__break', '__continue', '__for'
]

var_counter = 0

def freshName(): # for generating AST
    global var_counter
    n_var = var_counter
    var_counter += 1
    return "v" + str(n_var)

stFresh = 0
stBlock = []
stFun   = []
def run(f):
    global stFresh, stBlock, stFun
    sF = stFresh
    sB = stBlock
    sN = stFun
    try:
        return f()
    finally:
        stFresh = sF
        stBlock = sB
        stFun = sN

def fresh():
    global stFresh
    stFresh += 1
    return Rep("x"+str(stFresh-1))

def reify(f, *args):
    def f1():
        global stBlock
        stBlock = ['begin']
        try:
            last = f(*args)
            return stBlock + [ last ]
        except NonLocalReturnValue as e:
            raise NonLocalReturnValue(stBlock + [e.value]) # propagate exception ...
    return run(f1)

def reflect(s):
    global stBlock
    id = fresh()
    stBlock += [["let", id, s]]
    return id

def reflectTensor(args):
    rep = reflect(args)
    return RepTensor(rep.n)

def reflectTuple(args):
    rep = reflect(args)
    return RepTuple(rep.n)

class Rep(object):
    def __init__(self, n):
        self.n = n
    def __add__(self, m):
        return reflect(["+",self,m])
    def __sub__(self, m):
        return reflect(["-",self,m])
    def __mul__(self, m):
        return reflect(["*",self,m])
    def __rmul__(m, self):
        return reflect(["*",m,self])
    def __truediv__(self, m):
        return reflect(["/",self,m])
    def __mod__(self, m):
        return reflect(["%",self,m])
    def __eq__(self, m):
        return reflect(["==",self,m])
    def __ne__(self, m):
        return reflect(["!=",self,m])
    def __le__(self, m):
        return reflect(["<=",self,m])
    def __lt__(self, m):
        return reflect(["<",self,m])
    def __ge__(self, m):
        return reflect([">=",self,m])
    def __gt__(self, m):
        return reflect([">",self,m])
    def __repr__(self):
        return str(self.n)
    def __call__(self):
        return str(self.n)

class RepTensor(Rep):
    def __init__(self, n):
        super().__init__(n)
    def __add__(self, m):
        return reflectTensor(["+",self,m])
    def __mul__(self, m):
        return reflectTensor(["dot",self,m])
    def __truediv__(self, m):
        return reflectTensor(["/",self,m])
    def __getitem__(self, i):
        return reflectTensor(["idx",self,i])
    def __setitem__(self,i,v):
        return reflectTensor(["set-idx",self,i,v])

    @property
    def data(self):
        return reflectTensor(["getattr",self,"data"])

    @data.setter
    def data(self, v):
        return reflectTensor(["setattr",self,"data",v])

    def data_get(self, i):
        return reflectTensor(["array-get",self,"data",i])
    def data_set(self, i, v):
        return reflectTensor(["array-set",self,"data",i,v])
    def dot(self, t):
        return reflectTensor(["dot",self,t])
    def backward(self):
        return reflectTensor(["call",self,"backward"])
    def conv2d(self, kernel):
        return reflectTensor(["call",self,"conv2d",kernel])
    def view(self, *dims):
        return reflectTensor(["call",self,"view",dims])
    def print(self):
        return reflectTensor(["call",self,"print"])
    def max(self, n, keepDim=False):
        return reflectTensor(["call",self,"max",[n,keepDim]])
    def eq(self,m):
        return reflectTensor(["call",self,"eq"])
    def view_as(self,m):
        return reflectTensor(["call",self,"view_as",m])
    def sum(self):
        return reflectTensor(["call",self,"sum"])
    def sigmoid(self):
        return reflectTensor(["call",self,"sigmoid"])
    def tanh(self):
        return reflectTensor(["call",self,"tanh"])
    def exp(self):
        return reflectTensor(["call",self,"exp"])
    def log(self):
        return reflectTensor(["call",self,"log"])

stTensorFresh = 0

def freshTensor():
    global stTensorFresh
    stTensorFresh += 1
    return RepTensor("t"+str(stTensorFresh-1))

def newTensor(*dims):
    rep = reflect(["tensor", "[{}]".format(", ".join(list(map(str, dims))))])
    return RepTensor(rep.n)

class RepTuple(Rep):
    def __init__(self, n):
        super().__init__(n)
    @property
    def _1(self):
        return reflectTensor(["getattr",self,"_1"])
    @property
    def _2(self):
        return reflectTensor(["getattr",self,"_2"])
    @property
    def _3(self):
        return reflectTensor(["getattr",self,"_3"])

def rep_tuple(*args):
    tmp = reflectTuple(["call", "tuple", *args])
    return RepTuple(tmp.n)

class NonLocalReturnValue(Exception):
    def __init__(self, value):
        self.value = value

def __return(value):
    raise NonLocalReturnValue(value)

class NonLocalBreak(Exception): pass
class NonLocalContinue(Exception): pass

def __break(): raise NonLocalBreak()
def __continue(): raise NonLocalContinue()

def __print(value): # TODO HACK!
    if isinstance(value, RepTensor):
        return value.print()
    elif isinstance(value, str):
        return reflect(["print", '"{}"'.format(value)])
    else:
        return reflect(["print", value])

def reflectDef(name, args, f):
    global stBlock
    id = fresh()
    stBlock += [["def", name, args, f]]
    return id

def __def_staged(f, *args):
    import copy
    sig = inspect.signature(f)
    params = list(sig.parameters)
    nargs = []
    fargs = copy.deepcopy(args)
    if len(sig.parameters) is not len(args):
        raise NotImplemented
    for i in range(len(args)):
        if isinstance(args[i], Rep):
            nargs.append(params[i])
            fargs[i].n = params[i]

    return reflectDef(f.__name__, nargs, reify(lambda: f(*fargs))) # ['def', f.__name__, [*nargs], reify(lambda: f(*fargs))])

def __call_staged(f, *args):
    if f.__name__ is 'outputs':
        return reflectTensor([f.__name__, *args])
    return reflect([f.__name__, *args])

def __printf(s, vs):
    nvs = ['"{}"'.format(i) if isinstance(i, str) else '{}'.format(i) for i in vs]
    return reflect(["printf", ['"{}"'.format(s), "{}".format(", ".join(nvs))]])

def __var():
    return reflect(["new"])

def __assign(name, value):
    return reflect(["set", name, value])

def __read(name):
    if isinstance(name, RepTensor):
        return reflectTensor(["get", name])
    return reflect(["get", name])

def __len(name):
    return reflect(["len", name])

def __if(test, body, orelse):
    if isinstance(test, bool):
        if test:
            return body()
        else:
            return orelse()
    else:
        # There's a little bit of complication dealing with
        # __return: we currently require that either both
        # of the if branches __return, or none of them.
        def capture(f):
            try: return (False, reify(f))
            except NonLocalReturnValue as e:
                return (True, e.value)
        thenret, thenp = capture(body)
        # if len(thenp) > 1:
        #     thenp.insert(0, "begin")
        elseret, elsep = capture(orelse)
        # if len(elsep) > 1:
        #     elsep.insert(0, "begin")
        rval = reflect(["if", test, thenp, elsep])
        if thenret & elseret:
            raise NonLocalReturnValue(rval) # proper return
        elif (not thenret) & (not elseret):
            return rval
        else:
            raise Exception("if/else: branches must either both return or none of them")

def __while(test, body):
    # z = test()

    # if isinstance(z, bool):
    #     while z:
    #         try:
    #             body()
    #             z = test()
            #do other stuff

    if isinstance(test, bool): #test = x < 3
        while test: #is this evaluating correctly? I don't think it is -- might need to pass this as a function as well
            try: body()
            except NonLocalBreak as e:
                return None
            except NonLocalReturnValue as e:
                return e.value
            except NonLocalContinue as e:
                pass
        pass

    # We don't currently support return inside while
    def capture(f):
        try: return (False, reify(f))
        except NonLocalReturnValue as e:
            return (True, e.value)
        except NonLocalContinue as e:
            return (False, __while(test, f)) #f must be body if we hit a continue
    testret, testp = capture(test)
    bodyret, bodyp = capture(body)
    rval = reflect(["while", testp, bodyp])
    if (not testret) & (not bodyret):
        return rval
    else:
        raise Exception("while: return in body not allowed")

def __for(target, it, body):
    if isinstance(it, list):
        for target in it:
            try: body()
            except NonLocalBreak as e:
                return None
            except NonLocalReturnValue as e:
                return e.value
            except NonLocalContinue as e:
                pass
        pass

    def capture(f):
        try: return (False, reify(f))
        except NonLocalReturnValue as e:
            return e.value

    targetret, targetp = capture(target)
    itret, itp = capture(it)
    bodyret, bodyp = capture(body)
    rval = reflect(["for", targetp, itp, bodyp])
    if (not targetret) & (not bodyret):
        return rval
    else:
        raise Exception("for: return in body not allowed")
