from constants import *

def parseFunction(reader):
    fun_expr = ["def"]

    #read function name
    fun_expr.append(reader.getNextWord())

    # read arguments
    args = []
    reader.emitDELIMS()
    reader.acceptChar(OPEN_NODE)
    while(reader.peekChar() != CLOSE_NODE):
          args.append(reader.getNextWord())
    reader.acceptChar(CLOSE_NODE)
    fun_expr.append(args)

    # read body
    fun_expr.append(parseNode(reader))

    return fun_expr

def parseBegin(reader):
    # read body
    return parseNode(reader)

def parseLet(reader):
    let_expr = ["let"]

    # get var name
    let_expr.append(reader.getNextWord())

    # parse rhs
    rhs = parseNode(reader)

    # parse body
    body = parseNode(reader)

    let_expr.extend((rhs, body))

    return let_expr

def parseLiteral(reader):
    return reader.getNextWord()

def parseWhile(reader):
    while_expr = ["while"]
    
    #parse condition
    while_expr.append(parseNode(reader))

    #parse body
    while_expr.append(parseNode(reader))

    return while_expr


def parseArrayGet(reader):
    arget_expr = ["array-get"]

    #parse array var
    arget_expr.append(parseNode(reader))

    #parse index
    arget_expr.append(parseNode(reader))

    return arget_expr

def parseArraySet(reader):
    arset_expr = ["array-set"]

    #parse array var
    arset_expr.append(parseNode(reader))

    #parse index
    arset_expr.append(parseNode(reader))

    #parse rhs
    arset_expr.append(parseNode(reader))

    return arset_expr

def parseDot(reader):
    dot_expr = ["dot"]

    #parse var
    dot_expr.append(parseNode(reader))

    #parse attribute
    dot_expr.append(parseNode(reader))

    return dot_expr

def parseSet(reader):
    set_expr = ["set"]

    #parse name
    set_expr.append(parseNode(reader))

    #parse rhs
    set_expr.append(parseNode(reader))

    return set_expr

def parseGet(reader):
    get_expr = ["get"]
    
    #parse name
    get_expr.append(parseNode(reader))

    return get_expr

def parseLen(reader):
    len_expr = ["len"]
    
    #parse name
    len_expr.append(parseNode(reader))

    return len_expr

def parseIf(reader):
    if_expr = ["if"]
    
    #parse condition
    if_expr.append(parseNode(reader))

    #parse then
    if_expr.append(parseNode(reader))

    #parse else
    if_expr.append(parseNode(reader))

    return if_expr

def parseTensor(reader):
    tensor_expr = ["tensor"]

    #parse dims
    dims = []
    while(reader.peekChar() != CLOSE_NODE):
        dims.append(parseNode(reader))
    tensor_expr.append(dims)

    return tensor_expr

def parseTuple(reader):
    tuple_expr = ["tuple"]

    #parse tuple elements
    elems = []
    while(reader.peekChar() != CLOSE_NODE):
        elems.append(parseNode(reader))
    tuple_expr.append(elems)

    return tuple_expr


def parseCall(reader):
    call_expr = ["call"]

    #get function name
    call_expr.append(parseNode(reader))

    #parse args
    args = []
    while(reader.peekChar() != CLOSE_NODE):
        args.append(parseNode(reader))
    call_expr.append(args)

    return call_expr

def parsePrint(reader):
    print_expr = ["print"]

    #parse print string
    print_expr.append(parseNode(reader))

    return print_expr

def parsePrintf(reader):
    printf_expr = ["printf"]

    reader.acceptChar(OPEN_NODE)

    #parse print string
    printf_expr.append(parseNode(reader))

    #parse args
    args = []
    while(reader.peekChar() != CLOSE_NODE):
        args.append(parseNode(reader))
    reader.acceptChar(CLOSE_NODE)
    printf_expr.append(args)

    return printf_expr

def parseFor(reader):
    for_expr = ["for"]

    for_expr.append(parseNode(reader))

    #read "in"
    reader.getNextWord()

    #parse it
    for_expr.append(parseNode(reader))

    #parse body
    for_expr.append(parseNode(reader))

    return for_expr


def parseNew(reader):
    return ["new"]

def parseRet(reader):
    ret_expr = ["ret"]

    # parse return expression
    ret_expr.append(parseNode(reader))

    return ret_expr

def parseBinaryOp(reader, op):
    bop_expr = ["bop", op]

    # eval lhs
    bop_expr.append(parseNode(reader))

    #eval rhs
    bop_expr.append(parseNode(reader))

    return bop_expr

def parserNotImpl(reader):
    raise Exception("Parser not Implemented yet")


parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet,
    "call": parseCall,
    "while": parseWhile,
    "for": parseFor,
    "for_dataloader": "",
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
    "new": parseNew,
    "set": parseSet,
    "get": parseGet,
    "len": parseLen,
    "ret": parseRet,
}

binary_ops = {"+", "-", "*", "/", "%", "==", "!=", "<=", "<", ">=", ">"}

def parseNode(reader):
    reader.emitDELIMS()

    if(reader.peekChar() != OPEN_NODE):
        return parseLiteral(reader)

    reader.acceptChar(OPEN_NODE)

    # call respective parsers for keywords
    keyword = reader.getNextWord() 

    if keyword in binary_ops :
        gen = parseBinaryOp(reader, keyword)
    elif keyword in parsers:   
        gen = parsers[keyword](reader)
    else:
        raise Exception("Parser not implemented for `{}`\n".format(keyword))

    reader.acceptChar(CLOSE_NODE);

    # print(gen)
    return gen