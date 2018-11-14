from constants import *
from utils import *

var_counter = 0

def freshName(): # for generating functions
    global var_counter
    n_var = var_counter
    var_counter += 1
    return "genf" + str(n_var)

def getBeginBody(n):
    if isinstance(n, Node) and n.getNodeType() == "begin":
        return getBeginBody(n.getListArgs()[0])
    return n

def parseFunction(n):
    l = n.getListArgs()
    #currently args parsed as a node. resolve to a list of literals
    fargs = l[1].getListArgs()
    fargs.insert(0, Literal(l[1].getNodeType()))

    l_args = [l[0], fargs] #fname
    l_args.append(parseAST(getBeginBody(l[2]))) #skip begin in body

    if len(l) > 2:
        for i in range(3, len(l)):
            l_args.append(parseAST(l[i]))

    return Node("def", l_args)

def parseBegin(n):
    l = n.getListArgs()
    fname = freshName()
    body = parseAST(l[0])
    return Node("def", [Literal(fname), [], body])

def parseLet(n):
    l = n.getListArgs()
    # return body in let if a literal, and not NONE

    rhs = parseAST(l[1])
    body = parseAST(l[2])

    if isinstance(body, Literal):
        if body.getValue() != "None":
            body = Node("ret", [body])

    return Node("let", [l[0], rhs, body])

def parseWhile(n):
    l = n.getListArgs()

    # change condition to a function call
    cond_exp = parseAST(l[0]) # is a fundef
    cond_l = cond_exp.getListArgs()

    # body
    body_exp = parseAST(getBeginBody(l[1])) # remove begin

    call_cond = Node("call", [cond_l[0], []])

    while_exp = Node("while", [call_cond, body_exp])

    cond_fun = Node("def", [cond_l[0], cond_l[1], cond_l[2], while_exp])

    return cond_fun

def parseIf(n): 
    l = n.getListArgs()
    cond = parseAST(l[0])
    then_expr = parseAST(getBeginBody(l[1]))
    else_expr = parseAST(getBeginBody(l[2]))

    return Node(n.getNodeType(), [cond, then_expr, else_expr])

def parseFor(n):
    l = n.getListArgs()

    i = parseAST(l[0])
    # l[1] = "in"
    it = parseAST(l[2])
    body = parseAST(getBeginBody(l[3]))
    return Node(n.getNodeType(), [i, it, body])

def parseArrayGet(n):
    return n

def parseArraySet(n):
    l = n.getListArgs()
    return Node(n.getNodeType(), [l[0], parseAST(l[1])])

def parseDot(n):
    return n

def parseSet(n):
    l = n.getListArgs()
    return Node(n.getNodeType(), [l[0], parseAST(l[1])])

def parseGet(n):
    return n.getListArgs()[0] #return as literal

def parseLen(n):
    return n

def parseTensor(n):
    return n

def parseTuple(n):
    return n

def parseCall(n):
    # currently has args as a node, need to make it into a list of literals
    l = n.getListArgs()
    args_node = l[-1]
    args_list = [Literal(args_node.getNodeType())]

    for el in args_node.getListArgs():
        args_list.append(el)
    
    l[-1] = args_list
    return Node("call", l)

def parsePrint(n):
    return n

def parsePrintf(n):
    return n

def parseBinaryOp(n):
    l = n.getListArgs()
    op = n.getNodeType()
    l.insert(0, Literal(op))
    return Node("bop", l)

def parserNotImpl(n):
    raise Exception("Parser not Implemented yet")


parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet,
    "call": parseCall,
    "while": parseWhile,
    "for": parseFor,
    "for_dataloader": parserNotImpl,
    "if": parseIf,
    "array-get": parseArrayGet,
    "array-set": parseArraySet,
    "getattr": parserNotImpl,
    "setattr": parserNotImpl,
    "dot": parseDot,
    "tensor": parseTensor,
    "tuple": parseTuple,
    "print": parsePrint,
    "printf": parsePrintf,
    "set": parseSet,
    "get": parseGet,
    "len": parseLen,
}

binary_ops = {"+", "-", "*", "/", "%", "==", "!=", "<=", "<", ">=", ">"}

def parseAST(n):
    if isinstance(n, Literal):
        gen = n
    elif isinstance(n, Node):
        keyword = n.getNodeType()
        # eval 
        if keyword in binary_ops :
            gen = parseBinaryOp(n)
        elif keyword in parsers:   
            gen = parsers[keyword](n)
        else:
            raise Exception("Parser not implemented for `{}`\n".format(keyword))
    else:
        raise Exception("malformed expression, Expr not type Node or Literal")

    return gen