from constants import *

def parseFunction(genCode, reader):
    # read function name
    fname = reader.getNextWord()

    # read arguments
    reader.emitDELIMS()
    reader.acceptChar(OPEN_NODE)
    fargs = []
    while(reader.peekChar() != CLOSE_NODE):
        fargs.append(reader.getNextWord())
    reader.acceptChar(CLOSE_NODE)

    genCode.append("function {} ({}) (\n".format(fname, fargs))

    # read body
    parseNode(genCode, reader)

    genCode.append("\n)\n")

def parseBegin(genCode, reader):
    # read body
    body = parseNode(genCode, reader)
    if(body != None): 
        genCode.append("{}".format(body))

def parseLet(genCode, reader):
    # get var name
    varName = reader.getNextWord()
    genCode.append("{} = ".format(varName))

    # parse rhs
    parseNode(genCode, reader)
    genCode.append("\n")

    # parse body
    parseNode(genCode, reader)

def parseLiteral(genCode, reader):
    lit = reader.getNextWord()
    genCode.append(lit)

def parseWhile(genCode, reader):
    genCode.append("while (")
    
    #parse condition
    parseNode(genCode, reader)

    genCode.append(") (\n")

    #parse body
    parseNode(genCode, reader)

    genCode.append("\n)\n")

def parseArrayGet(genCode, reader):
    #parse array var
    parseNode(genCode, reader)

    genCode.append("[")

    #parse index
    parseNode(genCode, reader)

    genCode.append("]")

def parseArraySet(genCode, reader):
    #parse array var
    parseNode(genCode, reader)

    genCode.append("[")

    #parse index
    parseNode(genCode, reader)

    genCode.append("] = ")    

    #parse rhs
    parseNode(genCode, reader)

def parseDot(genCode, reader):
    #parse var
    parseNode(genCode, reader)

    genCode.append(".")

    #parse attribute
    parseNode(genCode, reader)

def parseSet(genCode, reader):
    #parse name
    parseNode(genCode, reader)

    genCode.append(" = ")

    #parse rhs
    parseNode(genCode, reader)

def parseGet(genCode, reader):
    #parse name
    parseNode(genCode, reader)

def parseLen(genCode, reader):
    genCode.append("len(")
    
    #parse name
    parseNode(genCode, reader)

    genCode.append(")")

def parseIf(genCode, reader):
    genCode.append("if (")
    
    #parse condition
    parseNode(genCode, reader)

    genCode.append(") (\n")

    #parse then
    parseNode(genCode, reader)

    genCode.append("else (\n")

    #parse else
    parseNode(genCode, reader)

    genCode.append("\n)\n")


def parseTensor(genCode, reader):
    #call torch.tensor
    genCode.append("torch.tensor(")

    #parse dims
    while(reader.peekChar() != CLOSE_NODE):
        parseNode(genCode, reader)
        genCode.append(", ")

    genCode.append(")")

def parseTuple(genCode, reader):
    genCode.append("(")

    #parse tuple elements
    while(reader.peekChar() != CLOSE_NODE):
        parseNode(genCode, reader)
        genCode.append(", ")

    genCode.append(")")

def parseCall(genCode, reader):
    #get function name
    fname = parseNode(genCode, reader)
    
    genCode.append("(")

    #parse args
    while(reader.peekChar() != CLOSE_NODE):
        parseNode(genCode, reader)
        genCode.append(", ")

    genCode.append(")")

def parsePrint(genCode, reader):
    genCode.append("print(")

    #parse print string
    parseNode(genCode, reader)

    genCode.append(")")

def parseNew(genCode, reader):
    genCode.append("None")

def parseRet(gencCode, reader):
    genCode.append("return ")

    # parse return expression
    parseNode(gencCode, reader)

def parseBinaryOp(genCode, reader, op):
    # eval lhs
    parseNode(genCode, reader)

    genCode.append(" {} ".format(op))

    #eval rhs
    parseNode(genCode, reader)

def parserNotImpl(genCode, reader):
    raise Exception("Parser not Implemented yet")


parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet,
    "call": parseCall,
    "while": parseWhile,
    "for": "",
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
    "printf": "",
    "new": parseNew,
    "set": parseSet,
    "get": parseGet,
    "len": parseLen,
    "ret": parseRet,
}

binary_ops = {"+", "-", "*", "/", "%", "==", "!=", "<=", "<", ">=", ">"}

def parseNode(genCode, reader):
    reader.emitDELIMS()

    if(reader.peekChar() != OPEN_NODE):
        return parseLiteral(genCode, reader)

    reader.acceptChar(OPEN_NODE)

    # call respective parsers for keywords
    keyword = reader.getNextWord() 

    if keyword in binary_ops :
        parseBinaryOp(genCode, reader, keyword)
    elif keyword in parsers:   
        parsers[keyword](genCode, reader)
    else:
        raise Exception("Parser not implemented for `{}`\n".format(keyword))

    reader.acceptChar(CLOSE_NODE);