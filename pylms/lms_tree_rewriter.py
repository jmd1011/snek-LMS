import sys
import ast
import types
import parser
import inspect
import builtins
import astunparse
import collections

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

        if isinstance(node.targets[0], ast.Attribute):
            self.generic_visit(node)
            return
        elif isinstance(node.targets[0], ast.Tuple):
            ids = list(map(lambda x: x.func.id if isinstance(x, ast.Call) else x.id, node.targets[0].elts))
        else:
            ids = [node.targets[0].id] # TODO: brittle, should look at shadowing, etc.

        locals = self.fundef.locals

        for id in ids:
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
    def __init__(self, scope):
        self.fundef = None # keep track of the current function we're in
        self.var_names = {}
        self.scope = scope
        self.recs = []
        super()

    def freshName(self,s = ""):
        if s not in self.var_names:
            self.var_names[s] = 0
        self.var_names[s] += 1
        return "{0}${1}".format(s, self.var_names[s])

    def shouldLiftVar(self, id):
        # lift a var if it's assigned more than once
        # TODO: need to check super scopes?
        # print('\nchecking {}'.format(astunparse.dump(self.fundef)))
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
        self.fundef = node.parent
        return new_node

    def visit_Assign(self, node):
        assert(len(node.targets) == 1) # FIXME (doesn't work -- if multiple targets, it's a Tuple (single))

        if isinstance(node.targets[0], ast.Attribute):
            self.generic_visit(node)
            return node
        elif isinstance(node.targets[0], ast.Tuple):
            self.generic_visit(node)
            return node

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

        if len(node.orelse) is 0:
            node.orelse = [ast.Pass()]

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
                                  decorator_list=[],
                                  returns=[])
        bFun = ast.FunctionDef(name=bFun_name,
                                  args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.body,
                                  decorator_list=[],
                                  returns=[])
        ast.fix_missing_locations(tFun)
        ast.fix_missing_locations(bFun)

        self.scope.visit(tFun)
        self.scope.visit(bFun)

        self.visit(tFun)
        self.visit(bFun)

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

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            if isinstance(node.func.value.value, ast.Attribute) and isinstance(node.func.value.value.value, ast.Name):
                if node.func.value.value.value.id is 'torch' and node.func.value.value.attr is 'utils' and node.func.value.attr is 'data' and node.func.attr is 'DataLoader':
                    args = [
                        ast.Str(s=node.args[0].func.attr), #set
                        node.args[0].keywords[0].value, #train
                        node.args[0].keywords[1].value, #download
                        node.args[0].keywords[2].value
                    ]
                    new_node = ast.Call(func=ast.Name(id='torch_loader', ctx=ast.Load()),
                                        args=args,
                                        keywords=[])
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node


        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id is 'torch':
                if node.func.attr is 'Tensor':
                    new_node = ast.Call(func=ast.Name(id='newTensor', ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

                else:
                    new_node = ast.Call(func=ast.Name(id='torch_{}'.format(node.func.attr), ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

            if node.func.value.id is 'nn':
                if node.func.attr is 'Linear':
                    new_node = ast.Call(func=ast.Name(id="nn_linear", ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

                if node.func.attr is 'Conv2d':
                    new_node = ast.Call(func=ast.Name(id="nn_conv2d", ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

            if node.func.value.id is 'transforms':
                if node.func.attr is 'Compose':
                    new_node = ast.Call(func=ast.Name(id='trans_compose', ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

                if node.func.attr is 'ToTensor':
                    new_node = ast.Call(func=ast.Name(id='trans_to_tensor', ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

                if node.func.attr is 'Normalize':
                    new_node = ast.Call(func=ast.Name(id='trans_normalize', ctx=ast.Load()),
                                        args=node.args,
                                        keywords=node.keywords)
                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

            # TODO(James): Fix this hacky nonsense
            if node.func.value.id in ['optim', 'F', 'onnx', 'lantern']:
                new_node = ast.Call(func=ast.Name(id='{}_{}'.format(node.func.value.id, node.func.attr), ctx=ast.Load()),
                                    args=node.args,
                                    keywords=node.keywords)
                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                return new_node

        if not isinstance(node.func, ast.Name):
            return node

        if self.fundef is not None and self.fundef.name == node.func.id:
            if node.func.id not in self.recs:
                self.recs += [node.func.id]
            new_node = ast.Call(func=ast.Name(id='__call_staged', ctx=ast.Load()),
                                args=[ast.Name(id=node.func.id, ctx=ast.Load())] + node.args,
                                keywords=node.keywords)
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        if node.func.id in self.recs:
            new_node = ast.Call(func=ast.Name(id='__call_staged', ctx=ast.Load()),
                                args=[ast.Name(id=node.func.id, ctx=ast.Load())] + node.args,
                                keywords=node.keywords)
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        if node.func.id is 'Variable':
            new_node = ast.Call(func=ast.Name(id='rep_variable', ctx=ast.Load()),
                                args=node.args,
                                keywords=node.keywords)
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        if node.func.id is 'print':
            if isinstance(node.args[0], ast.Call) and node.args[0].func.attr is 'format':
                args = [
                    node.args[0].func.value,
                    ast.List(elts=node.args[0].args,ctx=ast.Load())
                ]
                new_node = ast.Call(func=ast.Name(id='__printf', ctx=ast.Load()),
                                    args=args,
                                    keywords=[])

                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                return new_node

            else:
                new_node = ast.Call(func=ast.Name(id='__print', ctx=ast.Load()),
                                    args=node.args,
                                    keywords=[])

                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                return new_node

        if node.func.id is 'len':
            new_node = ast.Call(func=ast.Name(id='__len', ctx=ast.Load()),
                                args=node.args,
                                keywords=[])
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)

        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Call):
                if node.value.attr is 'data':
                # print("{}".format(ast.dump(node)))
                # if node.value.value.func.value.id is 'F' and node.value.value.attr is 'data':
                    new_node = ast.Call(func=ast.Attribute(value=node.value.value, attr='data_get', ctx=ast.Load()), args=[node.slice.value], keywords=[])
                    # new_node.value.func.attr = 'data_get'

                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node

            elif node.value.attr is 'data':
                new_node = ast.Call(func=ast.Attribute(value=node.value.value, attr='data_get', ctx=ast.Load()), args=[node.slice.value], keywords=[])
                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                return new_node

        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        new_node = ast.Expr(ast.Call(func=ast.Name(id='__return', ctx=ast.Load()),
                                                   args=[node.value],
                                                   keywords=[]))
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        mod = new_node

        if isinstance(node.value, ast.Call) and node.value.func.id is '__call_staged':
            def_node = ast.Expr(ast.Call(func=ast.Name(id='__def_staged', ctx=ast.Load()),
                                args=node.value.args,
                                keywords=node.value.keywords))

            ast.copy_location(def_node,new_node)
            ast.fix_missing_locations(def_node)
            mod = [def_node, new_node]
        return mod

    def visit_For(self, node):
        self.generic_visit(node)
        #| recognize the pattern of PyTorch's DataLoader |#
        def isPyTorchDataLoader(tgt, iter):
            return isinstance(tgt, ast.Tuple) and \
            len(tgt.elts) == 2 and \
            isinstance(tgt.elts[0], ast.Name) and \
            isinstance(tgt.elts[1], ast.Tuple) and \
            len(tgt.elts[1].elts) == 2 and \
            isinstance(tgt.elts[1].elts[0], ast.Name) and \
            isinstance(tgt.elts[1].elts[1], ast.Name) and \
            isinstance(iter, ast.Call) and \
            iter.func.id == 'enumerate' and \
            'loader' in iter.args[0].id

        #| Transforms the target names to list of strings |#
        def targetToList(tgt):
            def extract(x):
                if isinstance(x, ast.Name): return x.id
                elif isinstance(x, ast.Tuple): return targetToList(x.elts)
                else: raise NotImplementedError
            return list(map(extract, tgt))

        def targetToFlatList(tgt):
            res = []
            for item in targetToList(tgt):
                if isinstance(item, list): res.extend(item)
                else: res.append(item)
            return res

        if isPyTorchDataLoader(node.target, node.iter):
            outer_fun_name = self.freshName("forfunc")
            outer_fun = ast.FunctionDef(name=outer_fun_name,
                                        args=ast.arguments(args=list(map(lambda x: ast.arg(arg=x, annotation=None),
                                                                         targetToFlatList(node.target.elts))),
                                                           vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                        body=node.body,
                                        decorator_list=[],
                                        returns=[])
            ast.fix_missing_locations(outer_fun)

            self.scope.visit(outer_fun)
            self.visit(outer_fun)

            new_node = ast.Expr(ast.Call(func=ast.Name(id='__for_dataloader', ctx=ast.Load()),
                                         args=[node.iter.args[0],
                                               ast.Name(id=outer_fun_name, ctx=ast.Load())],
                                         keywords=[]))
            #ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return [outer_fun, new_node]
        else:
            bFun_name = self.freshName("body")
            bFun = ast.FunctionDef(name=bFun_name,
                                  args=[], # ast.arguments(args=[ast.arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]),
                                  body=node.body,
                                  decorator_list=[],
                                  returns=[])
            ast.fix_missing_locations(bFun)

            self.scope.visit(bFun)
            self.visit(bFun)

            new_node = ast.Expr(ast.Call(
                func=ast.Name(id='__for', ctx=ast.Load()),
                args=[node.target,
                      node.iter,
                      ast.Name(id=bFun_name, ctx=ast.Load())],
                keywords=[]))
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return [bFun, new_node]
