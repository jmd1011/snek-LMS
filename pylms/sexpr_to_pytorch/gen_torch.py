from utils import *

def getListLiterals(l):
        s_l = []
        for el in l:
            s_l.append(el.getValue())
        return s_l

class GenCodeFromAST:
    def __init__(self):
        self.genCode = GenCode()

    def addImports(self, imports):
        #imports must be a list
        for el in imports:
            self.genCode.append(el)
            self.genCode.newLine()
        self.genCode.newLine() #one more for good luck 

    def genCodeFromAst(self, ast):
        if isinstance(ast, Literal): #assuming either list or string
            self.genCode.append(ast.getValue())
            return
        
        if not isinstance(ast, Node):
            raise Exception("malformed ast: {}".format(ast))
        
        l = ast.getListArgs()

        if ast.getNodeType() == 'def':
            fname = l[0].getValue()

            # TODO don't hardcode it in
            if(fname == "lossFun"):
                self.genCode.append("@torch.jit.script")
                self.genCode.newLine()

            #function definition
            self.genCode.append("def {} ({})".format(fname, ",".join(getListLiterals(l[1]))))
            self.genCode.startNewScope()
            
            # body  
            self.genCodeFromAst(l[2])
            self.genCode.endScope()

            if len(l) > 3:
                for i in range(3, len(l)):
                    self.genCodeFromAst(l[i])
                    self.genCode.newLine()


        elif ast.getNodeType() == 'let':
            stmts = {'if', 'while', 'for', 'def'}

            if isinstance(l[1], Literal) or (l[1].getNodeType() not in stmts):
                self.genCode.append("{} = ".format(l[0].getValue())) # var name

            if isinstance(l[1], Literal) and l[1].getValue() == 'new':
                self.genCode.append('None')
            else: 
                self.genCodeFromAst(l[1]) # rhs
            self.genCode.newLine()

            # body 
            self.genCodeFromAst(l[2])

        elif ast.getNodeType() == 'if':
            self.genCode.append("if ")

            # condition
            self.genCodeFromAst(l[0])
            
            # then
            self.genCode.startNewScope()
            self.genCodeFromAst(l[1])
            self.genCode.endScope()

            # else
            self.genCode.append("else")
            self.genCode.startNewScope()
            self.genCodeFromAst(l[2])
            self.genCode.endScope()

        elif ast.getNodeType() == 'while':

            self.genCode.append("while ")
            self.genCodeFromAst(l[0])

            self.genCode.startNewScope()

            # body
            self.genCodeFromAst(l[1])
            self.genCode.endScope()

        elif ast.getNodeType() == "array-get":
            self.genCode.append("{}[".format(l[0])) # var name
            self.genCodeFromAst(l[1]) # index
            self.genCode.append("]")

        elif ast.getNodeType() == 'array-set':
            self.genCode.append("{}[".format(l[0])) # var name
            self.genCodeFromAst(l[1]) # index
            self.genCode.append("] = ")
            self.genCodeFromAst(l[2]) # rhs

        elif ast.getNodeType() == 'dot':
            self.genCodeFromAst(l[0])
            self.genCode.append(".")
            self.genCodeFromAst(l[1])

        elif ast.getNodeType() == 'set':
            self.genCodeFromAst(l[0])
            self.genCode.append(" = ")
            self.genCodeFromAst(l[1])

        elif ast.getNodeType() == 'len':
            self.genCode.append("len(")
            self.genCodeFromAst(l[0])
            self.genCode.append(")")

        elif ast.getNodeType() == 'tensor':
            self.genCode.append("torch.tensor(")
            self.genCode.append(",".join(getListLiterals(l[0])))
            self.genCode.append(")")

        elif ast.getNodeType() == 'tuple':
            self.genCode.append("(")
            self.genCode.append(",".join(getListLiterals(l[0])))
            self.genCode.append(")")

        elif ast.getNodeType() == 'call':
            fname = l[0].getValue()
            torch_funs = {"relu": "F.relu", "nll_loss": "F.nll_loss", "log_softmax": "F.log_softmax"}
            if fname in torch_funs:
                fname = torch_funs[fname]

            self.genCode.append("{}".format(fname))

            # append dots
            for i in range(1, len(l) - 1):
                self.genCode.append(".{}".format(l[i].getValue()))
            
            # write params
            self.genCode.append("(")
            self.genCode.append(",".join(getListLiterals(l[-1])))
            self.genCode.append(")")


        elif ast.getNodeType() == 'print':
            self.genCode.append("print(")
            self.genCodeFromAst(l[0])
            self.genCode.append(")")

        elif ast.getNodeType() == 'printf':
            self.genCode.append("print(")
            self.genCodeFromAst(l[0])
            self.genCode.append(").format(")
            self.genCode.append(",".join(getListLiterals(l[1])))
            self.genCode.append(")")

        elif ast.getNodeType() == 'for':
            self.genCode.append("for")
            self.genCodeFromAst(l[0])
            self.genCode.append(" in ")
            self.genCodeFromAst(l[1])
            self.genCode.startNewScope()
            self.genCodeFromAst(l[2])
            self.genCode.endScope()

        elif ast.getNodeType() == 'bop':
            self.genCodeFromAst(l[1]) 
            self.genCode.append(" {} ".format(l[0].getValue()))
            self.genCodeFromAst(l[2]) 
        
        elif ast.getNodeType() == 'ret':
            self.genCode.append("return {}".format(ast.getListArgs()[0].getValue()))

        else:
            raise Exception("Not found keyword for: {}".format(l[0]))

    def getGenCode(self):
        return self.genCode.getCode()