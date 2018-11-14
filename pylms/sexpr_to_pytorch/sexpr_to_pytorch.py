from utils import Reader
from parse_ast import parseAST
from gen_torch import GenCodeFromAST
from parse_sexp import *

TORCH_IMPORTS = [
    "import torch", 
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import torch.optim as optim",
]

def torchTheSnake(sexpr):
    if not isinstance(sexpr, str):
        raise Exception("argument is not a string")

    # initialize reader and generated code classes
    reader = Reader(sexpr)

    ast = parseSexp(reader)
    # printParsedSexp(ast)
    # print("\n")
    com_ast = parseAST(ast)
    # printParsedSexp(com_ast)
    # print("\n") 
    fromAst = GenCodeFromAST()
    fromAst.addImports(TORCH_IMPORTS)
    fromAst.genCodeFromAst(com_ast)
    print(fromAst.getGenCode())



# torchTheSnake("(def runWhile (in1) (begin (begin (let x7 new (let x8 (set x7 3) (let x9 (get x7) (let x10 (< x9 in1) (let x11 (while (begin (let x11 (get x7) (let x12 (< x11 in1) x12))) (begin (let x11 (get x7) (let x12 (+ x11 1) (let x13 (set x7 x12) None))))) (let x12 (get x7) x12)))))))))")
# torchTheSnake("(def runLift (in1) (begin (begin (let x7 new (let x8 (set x7 in1) (let x9 (get x7) (let x10 (> x9 0) (let x11 (if x10 (begin (let x11 (get x7) (let x12 (+ x11 1) (let x13 (set x7 x12) None)))) (begin (let x11 (get x7) (let x12 (- x11 1) (let x13 (set x7 x12) None))))) (let x12 (get x7) x12)))))))))")
# torchTheSnake("(def runX (in1) (begin (begin (let x0 new (let x1 (* in1 1) (let x2 (* in1 x1) (let x3 (* in1 x2) (let x4 (set x0 x3) (let x5 (get x0) x5)))))))))")
# torchTheSnake("(def runX (in1) (begin (begin (let x7 (* in1 1) (let x8 (* in1 x7) (let x9 (* in1 x8) x9))))))")
torchTheSnake("(def lossFun (x26 x27) (begin (let x28 (call x26 view (-1 784 )) (let x29 (call x5 (x28)) (let x30 (call relu (x29)) (let x31 (call x6 (x30)) (let x32 (call log_softmax (x31 dim=1)) (let x33 (call nll_loss (x32 x27 True)) x33))))))))")