import sys
import ast
import types
import parser
import inspect
import builtins
import astunparse


class ScopeAnalysis(ast.NodeVisitor):
    """
    Find single-assigment variables. These correspond to
    'val x = ...' in Scala and don't need to be lifted.
    """
    def __init__(self):
        self.fundef = None
        super()

    def visit_Assign(self, node):
        assert(len(node.targets) == 1) # FIXME
        id = node.targets[0].id # TODO: brittle, should look at shadowing, etc.

        locals = self.fundef.locals

        if not locals.get(id): locals[id] = 0
        locals[id] += 1

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.parent = self.fundef
        self.fundef = node
        node.locals = {}
        self.generic_visit(node)
        self.fundef = node.parent


class StagingRewriter(ast.NodeTransformer):
    """
    StagingRewriter does two things:
    1) virtualize primitives such as `if`, `while`, `for` and etc
    2) virtualize var accesses for non-single-assignment vars
    """
    def __init__(self):
        self.fundef = None # keep track of the current function we're in
        self.var_names = {}
        super()


    def freshName(self,s = ""):
        if s not in self.var_names:
            self.var_names[s] = 0
        self.var_names[s] += 1
        return "{0}${1}".format(s, self.var_names[s])

    def shouldLiftVar(self, id):
        # lift a var if it's assigned more than once
        # TODO: need to check super scopes?
        return ((self.fundef.locals.get(id)) and
               (self.fundef.locals[id] > 1))

    def visit_FunctionDef(self, node):
        node.parent = self.fundef
        self.fundef = node
        self.generic_visit(node)

        # generate code to pre-initialize staged vars
        # we stage all vars that are written to more than once
        inits = (ast.Assign(targets=[ast.Name(id=id, ctx=ast.Store())],
           value=ast.Call(func=ast.Name(id='__var', ctx=ast.Load()), args=[], keywords=[])) for id in node.locals if node.locals[id] > 1)

        new_node = ast.copy_location(ast.FunctionDef(name=node.name,
                                         args=node.args,
                                         body=[ast.Try(body=list(inits) + node.body,
                                                      handlers=[ast.ExceptHandler(type=ast.Name(id='NonLocalReturnValue', ctx=ast.Load()),
                                                                                       name='r',
                                                                                       body=[ast.Return(value=ast.Attribute(value=ast.Name(id='r', ctx=ast.Load()), attr='value', ctx=ast.Load()))])],
                                                      orelse=[],
                                                      finalbody=[])],
                                         decorator_list=list(filter(lambda n: n.id!='lms', node.decorator_list)),
                                         returns=node.returns),
                          node)
        ast.fix_missing_locations(new_node)
        # note: we're losing parent links and locals here. ok?
        # new_node.parent = node.parent # JD: there aren't parent links by default, so that's ok
        # new_node.locals = node.locals
        self.fundef = node.parent
        return new_node

    def visit_Assign(self, node):

        assert(len(node.targets) == 1) # FIXME
        id = node.targets[0].id

        # NOTE: grab id before -- recursive call will replace lhs with __read!!
        self.generic_visit(node)

        if not self.shouldLiftVar(id):
            return node

        new_node = ast.Expr(ast.Call(
            func=ast.Name(id='__assign', ctx=ast.Load()),
            args=[ast.Name(id=id, ctx=ast.Load()),
                  node.value
                 ],
            keywords=[]
        ))
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        return [new_node]

    def visit_Name(self, node):
        self.generic_visit(node)

        if not self.shouldLiftVar(node.id):
            return node

        new_node = ast.Call(
            func=ast.Name(id='__read', ctx=ast.Load()),
            args=[ast.Name(id=node.id, ctx=ast.Load())],
            keywords=[]
        )
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        return new_node

    def visit_If(self, node):
        self.generic_visit(node)
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

        new_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='__if', ctx=ast.Load()),
            args=[node.test,
                  ast.Name(id=tBranch_name, ctx=ast.Load()),
                  ast.Name(id=eBranch_name, ctx=ast.Load())
                 ],
            keywords=[]
        ))

        ast.fix_missing_locations(new_node)
        mod = [tBranch, eBranch, new_node]
        return mod

    def visit_While(self, node):
        self.generic_visit(node)

        tFun_name = self.freshName("cond")
        bFun_name = self.freshName("body")
        tFun = ast.FunctionDef(name=tFun_name,
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=[ast.Return(node.test)],
                                  decorator_list=[])
        bFun = ast.FunctionDef(name=bFun_name,
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.body,
                                  decorator_list=[])
        ast.fix_missing_locations(tFun)
        ast.fix_missing_locations(bFun)

        new_node = ast.Expr(ast.Call(
            func=ast.Name(id='__while', ctx=ast.Load()),
            args=[ast.Name(id=tFun_name, ctx=ast.Load()),
                  ast.Name(id=bFun_name, ctx=ast.Load()),
                 ],
            keywords=[]
        ))

        ast.fix_missing_locations(new_node)
        mod = [tFun, bFun, new_node]
        return mod

    def visit_Continue(self, node):
      self.generic_visit(node)

      new_node = ast.Expr(ast.Call(
          func=ast.Name(id='__continue', ctx=ast.Load()),
          args=[],
          keywords=[]
      ))

      ast.fix_missing_locations(new_node)
      return new_node

    def visit_Break(self, node):
      self.generic_visit(node)

      new_node = ast.Expr(ast.Call(
          func=ast.Name(id='__break', ctx=ast.Load()),
          args=[],
          keywords=[]
      ))

      ast.fix_missing_locations(new_node)
      return new_node

    def visit_Call(self, node):
        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return node

        if not node.func.id == 'print':
            return node

        new_node = ast.Call(func=ast.Name(id='__print', ctx=ast.Load()),
                                          args=node.args,
                                          keywords=[])
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node

    def visit_Return(self, node):
        self.generic_visit(node)
        new_node = ast.Expr(ast.Call(func=ast.Name(id='__return', ctx=ast.Load()),
                                          args=[node.value],
                                          keywords=[]))
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node

