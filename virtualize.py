import ast

class ConditionRepChecker(ast.NodeVisitor):
    def __init__(self, reps):
        self.reps = reps
        self.hasRep = False
        super()

    def visit_Name(self, node):
        print("-------Name-------")
        print(ast.dump(node))
        if node.id == "RepString" or node.id == "RepInt":
            self.hasRep = True

    def visit_BoolOp(self, node):
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)

def vIf(cond, tBranch, eBranch, reps):
    print("----------IF----------")
    x = ConditionRepChecker(reps)
    x.visit(cond)
    
    # x.hasRep Bool says whether or not node has a bool
    print("RESULT = " + str(x.hasRep))


    print("--------End IF--------")