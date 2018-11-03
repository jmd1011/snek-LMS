# for_dataloader
# while
# begin => 
# if
# for in

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
    genTorch(genCode, reader)

    #TODO: Fix append to genCode
    genCode.append("\n)\n")

def parseBegin(genCode, reader):
    # read body
    body = genTorch(genCode, reader)
    if(body != None): 
        genCode.append("{}".format(body))

def parseLet(genCode, reader):
    # get var name
    varName = reader.getNextWord()
    genCode.append("{} = ".format(varName))

    # parse rhs
    genTorch(genCode, reader)
    genCode.append("\n")

    # parse body
    genTorch(genCode, reader)

def parseLiteral(genCode, reader):
    lit = reader.getNextWord()
    genCode.append(lit + " ")

parsers = {
    "def": parseFunction,
    "begin": parseBegin,
    "let": parseLet
}

def genTorch(genCode, reader):
    reader.emitDELIMS()

    if(reader.peekChar() != OPEN_NODE):
        return parseLiteral(genCode, reader)

    reader.acceptChar(OPEN_NODE)

    # call respective parsers for keywords
    keyword = reader.getNextWord()    
    parsers[keyword](genCode, reader)

    reader.acceptChar(CLOSE_NODE);