import sys
import ast
import types
import parser
import inspect
import builtins
import astunparse

class StagingRewriter(ast.NodeTransformer):
    """
    StagingRewriter does two things:
    1) lift next-stage variables to be Rep
    2) virtualize primitives such as `if`, `while`, `for` and etc
    Note: Probably we may also rewrite operators such as `+` rather than overloading them.
    """
    def __init__(self, reps = {}):
        self.reps = reps
        super()

    var_names = {}

    def freshName(self,s = ""):
        if s not in self.var_names:
            self.var_names[s] = 0
        self.var_names[s] += 1
        return "{0}${1}".format(s, self.var_names[s])

    def visit_If(self, node):
        # TODO: Virtualization of `if`
        # If the condition part relies on a staged value, then it should be virtualized.
        # print(ast.dump(node))

        # if node is of the form (test, body, orelse)

        # iter_fields lets me iterate through the contents of the if node
        # gives the child as a tuple of the form (child-type, object)
        # cond_node = node.test;
        # check for BoolOp and then Compare
        # node.body = list(map(lambda x: self.generic_visit(x), node.body))
        self.generic_visit(node)
        # vIf(node.test, node.body, node.orelse, self.reps)
        tBranch_name = self.freshName("then")
        eBranch_name = self.freshName("else")
        tBranch = ast.FunctionDef(name=tBranch_name,
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.body,
                                  decorator_list=[])
        eBranch = ast.FunctionDef(name=eBranch_name,
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.orelse,
                                  decorator_list=[])
        ast.fix_missing_locations(tBranch)
        ast.fix_missing_locations(eBranch)

        self.generic_visit(tBranch)
        self.generic_visit(eBranch)

        # self.vIf(node.test, tBranch, eBranch)
        # print(node.lineno)
        # new_node = ast.Expr(ast.Call(func=ast.Name(id='__return', ctx=ast.Load()), args=[
        #     ast.Call(
        #     func=ast.Name(id='vIf', ctx=ast.Load()),
        #     args=[node.test,
        #           ast.Name(id=tBranch_name, ctx=ast.Load()),
        #           ast.Name(id=eBranch_name, ctx=ast.Load()),
        #           ast.Dict(list(map(ast.Str, self.reps.keys())),
        #                    list(map(ast.Str, self.reps.values()))),
        #          ],
        #     keywords=[]
        # )], keywords=[]))

        new_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='__if', ctx=ast.Load()),
            args=[node.test,
                  ast.Name(id=tBranch_name, ctx=ast.Load()),
                  ast.Name(id=eBranch_name, ctx=ast.Load()),
                  ast.Dict(list(map(ast.Str, self.reps.keys())),
                           list(map(ast.Str, self.reps.values()))),
                 ],
            keywords=[]
        ))

        ast.fix_missing_locations(new_node)
        # self.generic_visit(new_node)
        mod = [tBranch, eBranch, new_node]
        # print(ast.dump(eBranch))
        return mod
        #return ast.copy_location(mod, node)

    def visit_While(self, node):
        # TODO: Virtualization of `while`
        self.generic_visit(node)
        return node # COMMENT WHEN __while IS DONE

        # UNCOMMENT WHEN __while IS DONE
        # nnode = ast.copy_location(ast.Call(func=ast.Name('__while', ast.Load()),
        #                                 args=[node.test, node.body],
        #                                 keywords=[]),
        #                         node)
        # nnode.parent = node.parent
        # return nnode

    def visit_FunctionDef(self, node):
        #self.reps = getFunRepAnno(node)
        self.generic_visit(node)
        new_node = ast.copy_location(ast.FunctionDef(name=node.name,
                                         args=node.args,
                                         body=[ast.Try(body=node.body,
                                                      handlers=[ast.ExceptHandler(type=ast.Name(id='NonLocalReturnValue', ctx=ast.Load()), 
                                                                                       name='r', 
                                                                                       body=[ast.Return(value=ast.Attribute(value=ast.Name(id='r', ctx=ast.Load()), attr='value', ctx=ast.Load()))])],
                                                      orelse=[],
                                                      finalbody=[])],
                                         decorator_list=[],
                                         returns=node.returns),
                          node)
        ast.fix_missing_locations(new_node)
        #node.decorator_list = [] # Drop the decorator
        # print(node.name)
        return new_node

    def visit_Return(self, node):
        self.generic_visit(node)
        new_node = ast.Raise(exc=ast.Call(func=ast.Name(id='NonLocalReturnValue', ctx=ast.Load()),
                                          args=[node.value],
                                          keywords=[]),
                             cause=None)
        ast.fix_missing_locations(new_node)
        return new_node

        # ret_name = freshName("ret")
        # retfun = ast.FunctionDef(name=ret_name,
        #                           args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
        #                           body=[node],
        #                           decorator_list=[])

        # ast.fix_missing_locations(retfun)
        # retnode = ast.Expr(ast.Call(func=ast.Name('__return', ast.Load()),
        #                                 args=[ast.Name(id=ret_name, ctx=ast.Load())],
        #                                 keywords=[]))
        # ast.fix_missing_locations(retnode)
        # # ast.copy_location(retnode, node)
        # return [retfun, retnode]
        # return retnode

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id in self.reps:
            nnode = ast.copy_location(ast.Call(func=ast.Name(id=self.reps[node.id], ctx=ast.Load()),
                                              args=[ast.Str(s=node.id)],
                                              keywords=[]),
                                    node)
            return nnode
        return node
