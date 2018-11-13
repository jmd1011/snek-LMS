from utils import *
from gen_torch import parseAST
from parse_sexp import *

def torchTheSnake(sexpr):
    if not isinstance(sexpr, str):
        raise Exception("argument is not a string")

    # initialize reader and generated code classes
    reader = Reader(sexpr)

    ast = parseSexp(reader)
    printParsedSexp(ast)
    print("\n")
    com_ast = parseAST(ast)
    printParsedSexp(com_ast)
    print("\n") 
    fromAst = GenCodeFromAST()
    fromAst.genCodeFromAst(com_ast)
    print(fromAst.getGenCode())

def getListLiterals(l):
        s_l = []
        for el in l:
            s_l.append(el.getValue())
        return s_l

class GenCodeFromAST:
    def __init__(self):
        self.genCode = GenCode()

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
            if(fname == "lossfun"):
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
            torch_funs = {"relu": "F.relu", "nn_loss": "F.nn_loss", "log_softmax": "F.log_softmax"}
            if fname in torch_funs:
                fname = torch_funs[fname]

            self.genCode.append("{}(".format(fname))
            self.genCode.append(",".join(getListLiterals(l[1])))
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

# torchTheSnake("(def fname (in1) (begin a))")
# torchTheSnake("(let a b c)")
# torchTheSnake("(def f () (begin a))")
# torchTheSnake("(while (< a b) (begin a))")
# torchTheSnake("(array-get a 1)")
# torchTheSnake("(array-set a 1 (+ 3 5))")
# torchTheSnake("(dot a v)")
# torchTheSnake("(set a 1)")
# torchTheSnake("(print \"word\")")
# torchTheSnake("(printf (\"{}\" i))")
# torchTheSnake("(def runWhile (in1) (begin (begin (let x7 new (let x8 (set x7 3) (let x9 (get x7) (let x10 (< x9 in1) (let x11 (while (begin (let x11 (get x7) (let x12 (< x11 in1) x12))) (begin (let x11 (get x7) (let x12 (+ x11 1) (let x13 (set x7 x12) None))))) (let x12 (get x7) x12)))))))))")
# torchTheSnake("(def runLift (in1) (begin (begin (let x7 new (let x8 (set x7 in1) (let x9 (get x7) (let x10 (> x9 0) (let x11 (if x10 (begin (let x11 (get x7) (let x12 (+ x11 1) (let x13 (set x7 x12) None)))) (begin (let x11 (get x7) (let x12 (- x11 1) (let x13 (set x7 x12) None))))) (let x12 (get x7) x12)))))))))")
# torchTheSnake("(def runX (in1) (begin (begin (let x0 new (let x1 (* in1 1) (let x2 (* in1 x1) (let x3 (* in1 x2) (let x4 (set x0 x3) (let x5 (get x0) x5)))))))))")
torchTheSnake("(def lossfun () (begin (call nn_loss )) ")