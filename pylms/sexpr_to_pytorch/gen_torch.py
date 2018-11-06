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

    #TODO: Fix append to genCode
    genCode.append("function {} ({}) (\n".format(fname, fargs))

    # read body
    parseNode(genCode, reader)

    #TODO: Fix append to genCode
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

def parseNew(genCode, reader):
    # TODO: Check with James
    genCode.append("None")

def parseBinaryOp(genCode, reader, op):
    # eval lhs
    parseNode(genCode, reader)

    genCode.append(" {} ".format(op))

    #eval rhs
    parseNode(genCode, reader)


parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet,
    "call": "",
    "while": parseWhile,
    "for": "",
    "for_dataloader": "",
    "if": parseIf,
    "array-get": parseArrayGet,
    "array-set": parseArraySet,
    "getattr": "",
    "setattr": "",
    "dot": parseDot,
    "tensor": "",
    "tuple": "",
    "print": "",
    "printf": "",
    "new": parseNew,
    "set": parseSet,
    "get": parseGet,
    "len": parseLen,
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