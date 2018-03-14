__all__ = [
    'reflect', 'reify', 'fresh', 'Rep', 'NonLocalReturnValue', 'NonLocalBreak', 'NonLocalContinue',
    '__if', '__while', '__return', '__print',
    '__var', '__assign', '__read', '__break', '__continue'
]

var_counter = 0

def freshName(): # for generate AST
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

def reify(f):
    def f1():
        global stBlock
        stBlock = []
        try:
            last = f()
            return stBlock + [ last ]
        except NonLocalReturnValue as e:
            raise NonLocalReturnValue(stBlock + [e.value]) # propagate exception ...
    return run(f1)

def reflect(s):
    global stBlock
    id = fresh()
    stBlock += [["let", id, s]]
    return id



class Rep(object):
    def __init__(self, n):
        self.n = n
    def __add__(self, m):
        return reflect(["+",self,m])
    def __sub__(self, m):
        return reflect(["-",self,m])
    def __mul__(self, m):
        return reflect(["*",self,m])
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
    return reflect(["print", '"{}"'.format(value)])

def __var():
    return reflect(["new"])

def __assign(name, value):
    return reflect(["set", name, value])

def __read(name):
    return reflect(["get", name])

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
        if len(thenp) > 1:
            thenp.insert(0, "begin")
        elseret, elsep = capture(orelse)
        if len(elsep) > 1:
            elsep.insert(0, "begin")
        rval = reflect(["if", test, thenp, elsep])
        if thenret & elseret:
            raise NonLocalReturnValue(rval) # proper return
        elif (not thenret) & (not elseret):
            return rval
        else:
            raise Exception("if/else: branches must either both return or none of them")

def __while(test, body):
    if isinstance(test, bool):
        while test: #is this evaluating correctly? I don't think it is -- might need to pass this as a function as well
            try: body()
            except NonLocalBreak as e: #Need to add this
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
    testret, testp = capture(test)
    bodyret, bodyp = capture(body)
    rval = reflect(["while", testp, bodyp])
    if (not testret) & (not bodyret):
        return rval
    else:
        raise Exception("while: return in body not allowed")

