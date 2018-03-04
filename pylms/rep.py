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

def __if(test, body, orelse, reps):
    if(isinstance(test, bool)):
        if(test):
            res = body()
            return res
        else:
            res = orelse()
            return res
    else:
        # fixme: use irif?
        return irif(test, body, orelse)
	
