__all__ = ['reflect', 'Rep', 'NonLocalReturnValue', '__if', '__return']

def reflect(s):
    return Rep(s)

class Rep(object): 
    def __init__(self, n):
        self.n = n
    def __add__(self, m):
        return reflect(["+",self,m])
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
    def __repr__(self):
        return str(self.n)

class NonLocalReturnValue(Exception):
    def __init__(self, value):
        self.value = value

def __return(value):
    raise NonLocalReturnValue(value)

def __if(test, body, orelse):
    if(isinstance(test, bool)):
        if(test):
            return body()
        else:
            return orelse()
    else:
        # There's a little bit of complication dealing with
        # __return: we currently require that either both
        # of the if branches __return, or none of them.
        def capture(f):
            try: return (False, f())
            except NonLocalReturnValue as e:
                return (True, e.value)
        thenret, thenp = capture(body)
        elseret, elsep = capture(orelse)
        rval = reflect(["if", test, thenp, elsep])
        if thenret & elseret:
            raise NonLocalReturnValue(rval) # proper return
        elif (not thenret) & (not elseret):
            return rval
        else:
            raise Exception("if/else: branches must either both return or none of them")
	
