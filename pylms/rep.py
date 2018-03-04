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

def vIf(test, body, orelse, reps):
    # print("----------IF----------")
    if(isinstance(test, bool)):
        # print("No rep")
        if(test):
            # print(ast.dump(body()))
            # print("True")
            res = body()
            return res
        else:
            # print(type(orelse))
            # print(ast.dump(orelse()))
            # print("False")
            # print(orelse())
            res = orelse()
            return res
            # return orelse()
    else:
        return IRIf(test, body, orelse)
    # print("--------End IF--------")
	