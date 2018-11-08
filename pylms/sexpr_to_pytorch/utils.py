from constants import *

class GenCode:
    s = ""
    indent = ""

    def append(self, code):
        self.s += code
    
    def newLine(self):
        self.s += "\n" + self.indent

    def startNewScope(self):
        self.s += ":"
        self.indent += "  "
        self.newLine()

    def endScope(self):
        self.s += "\n"
        self.indent = self.indent[0: len(self.indent) - 2]

    def display(self):
        print(self.s)


class Reader:
        def __init__(self, str):
            self.str = str
            self.i = 0

        def acceptChar(self, c):
            # throw exception if character not found
            if(self.str[self.i] != c):
                raise Exception(
                    "expected: `{}` found: `{}` at index: {}\n"
                    .format(c, self.str[self.i], self.i))

            self.i += 1
        
        def peekChar(self):
            return self.str[self.i]

        def getNextWord(self):
            self.emitDELIMS()
            
            if(self.peekChar() == "\""):
                return self.readQuotedWord()

            # find next space or CLOSE_NODE after index i and return word
            idx_space = self.str.find(DELIM, self.i)
            idx_close = self.str.find(CLOSE_NODE, self.i)

            if(idx_close == -1 and idx_space == -1):
                raise Exception("Malformed expression")
                
            #absorb space but not CLOSE_NODE
            if(idx_close < idx_space or idx_space == -1): 
                word = self.str[self.i: idx_close]
                self.i = idx_close
            else:
                word = self.str[self.i: idx_space]
                self.i = idx_space + 1
            
            return word
        
        def peekNextWord(self):
            self.emitDELIMS()

            # find next space or CLOSE_NODE after index i and return word
            idx_space = self.str.find(DELIM, self.i)
            idx_close = self.str.find(CLOSE_NODE, self.i)

            #absorb space but not CLOSE_NODE
            if(idx_close < idx_space or idx_close != -1): 
                word = self.str[self.i: idx_close]
            else:
                word = self.str[self.i: idx_space]
            
            return word
        
        def readQuotedWord(self):
            self.emitDELIMS()

            if(self.peekChar() != "\""):
                raise Exception("Not a quoted word from index {}".format(self.i))
            
            idx_end = self.str.find("\"", self.i + 1)
            
            if(idx_end == -1):
                raise Exception("no ending quote found for starting index {}".format(self.i))

            word = self.str[self.i + 1: idx_end]
            self.i = idx_end + 1

            return word

        def emitDELIMS(self):
            while(self.str[self.i] == DELIM):
                self.i += 1

        def getIndex(self):
            return self.i