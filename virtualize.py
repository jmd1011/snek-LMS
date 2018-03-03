import ast
import astunparse

var_counter = 0

def freshName():
    global var_counter
    n_var = var_counter
    var_counter += 1
    return "1_$vb" + str(n_var)

class ConditionRepChecker(ast.NodeVisitor):
    def __init__(self, reps):
        self.reps = reps
        self.hasRep = False
        super()

    def visit_Name(self, node):
        print("-------Name-------")
        if node.id == "RepString" or node.id == "RepInt":
            self.hasRep = True

def vIf(test, body, orelse, reps):
    # print("----------IF----------")
    # x = ConditionRepChecker(reps)
    # x.visit(test)
    
    # # x.hasRep Bool says whether or not node has a bool
    # print("RESULT = " + str(x.hasRep))

    # if x.hasRep:
    #     # do stuff for rep here
    #     print("REP No EVAL")
    # else:
    #     # print("++++++++++++++++++\n")
    #     # print(ast.dump(node))
    #     # print("\n")
    #     # print(astunparse.dump(node))
    #     # print("\n++++++++++++++++++")
    #     ast.fix_missing_locations(test)
    #     c = compile(ast.Module(body=[test]), '<ast>', 'exec')
    #     exec(c, globals())
    #     # if eval(compile(cond, filename="<ast>", mode="exec"), globals()):
    #     #     eval(compile(tBranch, filename="<ast>", mode="exec"), globals())
    #     # else: 
    #     #     eval(compile(eBranch, filename="<ast>", mode="exec"), globals())
    print("--------End IF--------")