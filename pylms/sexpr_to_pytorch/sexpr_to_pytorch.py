from utils import *
from gen_torch import parseNode

def torchTheSnake(sexpr):
    if not isinstance(sexpr, str):
        raise Exception("argument is not a string")

    # initialize reader and generated code classes
    genCode = GenCode()
    reader = Reader(sexpr)

    parseNode(genCode, reader)
    genCode.display()

torchTheSnake("(def fname (in1) (begin (ret 1)))")
torchTheSnake("(let a b c)")
torchTheSnake("(def f () (begin a))")
torchTheSnake("(while (< a b) (begin a))")
torchTheSnake("(array-get a 1)")
torchTheSnake("(array-set a 1 (+ 3 5))")
torchTheSnake("(dot a v)")
torchTheSnake("(set a 1)")
torchTheSnake("(print \"word\")")
torchTheSnake("(printf (\"{}\" i))")
torchTheSnake("(for x9 in x7 (begin (let x10 (get x7) (let x11 (+ x10 1) (let x12 (set x7 x11) (let x13 (printf (\"{}\" x9)) None))))))")