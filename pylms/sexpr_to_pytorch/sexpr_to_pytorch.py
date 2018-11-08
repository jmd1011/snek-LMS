from utils import *
from gen_torch import parseNode

def torchTheSnake(sexpr):
    if not isinstance(sexpr, str):
        raise Exception("argument is not a string")

    # initialize reader and generated code classes
    genCode = GenCode()
    reader = Reader(sexpr)

    ast = parseNode(reader)
    print(ast)


torchTheSnake("(def fname (in1) (begin a))")
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

torchTheSnake("(def runLift (in1) (begin (begin (let x7 new (let x8 (set x7 in1) (let x9 (get x7) (let x10 (> x9 0) (let x11 (if x10 (begin (let x11 (get x7) (let x12 (+ x11 1) (let x13 (set x7 x12) None)))) (begin (let x11 (get x7) (let x12 (- x11 1) (let x13 (set x7 x12) None))))) (let x12 (get x7) x12)))))))))")