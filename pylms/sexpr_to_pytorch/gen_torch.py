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
    genCode.append(lit + " ")

def parseWhile(genCode, reader):
    return
    #parse condition

parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet,
    "call": "",
    "while": "",
    "for": "",
    "for_dataloader": "",
    "if": "",
    "array-get": "",
    "array-set": "",
    "getattr": "",
    "setattr": "",
    "dot": "",
    "tensor": "",
    "tuple": "",
    "print": "",
    "printf": "",
    "new": "",
    "set": "",
    "get": "",
    "len": "",
}

ops = {"+", "-", "*", "/", "%", "==", "!=", "<=", "<", ">=", ">"}

def parseNode(genCode, reader):
    reader.emitDELIMS()

    if(reader.peekChar() != OPEN_NODE):
        return parseLiteral(genCode, reader)

    reader.acceptChar(OPEN_NODE)

    # call respective parsers for keywords
    keyword = reader.getNextWord() 

    if(keyword not in parsers):
        raise Exception("Parser not implemented for `{}`\n".format(keyword))   
    parsers[keyword](genCode, reader)

    reader.acceptChar(CLOSE_NODE);