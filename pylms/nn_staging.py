# from pylms import lms, lmscompile, stage, ast
from .rep import *
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['nn_linear', 'nn_conv2d', 'newTensor', 'RepTensor', 'optim_SGD', 'F_nll_loss', '__variable', 'rep_train_loader_tensor', 'rep_train_loader_fresh', '__for_dataloader', 'F_relu', 'F_dropout', 'F_max_pool2d', 'F_log_softmax']

stFresh = 0

def freshTensor():
    global stFresh
    stFresh += 1
    return RepTensor("t"+str(stFresh-1))

def newTensor(*dims):
    rep = reflect(["tensor", "[{}]".format(", ".join(list(map(str, dims))))])
    return RepTensor(rep.n)

def reflectTensor(args):
    rep = reflect(args)
    return RepTensor(rep.n)

class RepTensor(Rep):
    def __init__(self, n):
        super().__init__(n)
    def __add__(self, m):
        return reflectTensor(["+",self,m])
    def __mul__(self, m):
        return reflectTensor(["dot",self,m])

    def data_get(self, i):
        return reflectTensor(["array-get",self,"data",i]);
    def backward(self):
        return reflectTensor(["call",self,"backward"]);
    def conv2d(self, kernel):
        return reflectTensor(["call",self,"conv2d",kernel.n]);
    def view(self, *dims):
        return reflectTensor(["call",self,"view",dims]);
    def print(self):
        return reflectTensor(["call",self,"print"])

def rep_train_loader_tensor():
    rept = reflect(freshTensor(None))
    return rept

def rep_train_loader_fresh():
    return fresh()


def nn_linear(hlsize, outsize):
    class Linear(object):
        def __init__(self):
            self.weight = newTensor(outsize, hlsize)
            self.bias =  newTensor(outsize)
            self.linear = None

        def __call__(self, tensor):
            if isinstance(tensor, torch.Tensor): #unstaged
                if self.linear is None:
                    self.linear = nn.Linear(hlsize, outsize)

                return self.linear(tensor)
            else: #staged
                return self.weight * tensor + self.bias

    return Linear()

def nn_conv2d(outSize, inSize, kernel_size, bias):
    class Conv2d(object):
        def __init__(self):
            self.kernel = newTensor(outSize, inSize, kernel_size, kernel_size)
            self.cond2d = None

        def __call__(self, tensor):
            if isinstance(tensor, torch.Tensor): #unstaged
                if self.cond2d is None:
                    self.conv2d = nn.Conv2d(hlsize, outsize, kernel_size=kernel_size, bias=bias)

                return self.conv2d(tensor)
            else: #staged
                return tensor.conv2d(self.kernel)
    return Conv2d()

def optim_SGD(params, lr, momentum):
    class RepSGD(Rep):
        def __init__(self, n):
            super().__init__(n)
            if isinstance(params, list):
                self.staged = False
                self.optim = optim.SGD(params, lr, momentum)
            else:
                self.staged = True
                self.optim = reflect([self, [lr, momentum]])

        def zero_grad(self):
            if self.staged:
                return reflectTensor(["call",self,"zero_grad"])
            else:
                return self.optim.zero_grad()

        def step(self):
            if self.staged:
                return reflectTensor(["call",self,"step"])
            else:
                return self.optim.step()

    return RepSGD("SGD")

def F_nll_loss(output, target, size_average=True):
    if isinstance(output, Variable):
        return F.nll_loss(output, target, size_average)
    else:
        return reflectTensor(["call", "nll_loss", [output, target, size_average]])

def F_relu(tensor):
    if isinstance(tensor, torch.Tensor):
        return F.relu(tensor)
    else:
        return reflectTensor(["call", "relu", [tensor]])

def F_dropout(tensor):
    if isinstance(tensor, torch.Tensor):
        return F.dropout(tensor)
    else:
        return reflectTensor(["call", "dropout", [tensor]])

def F_max_pool2d(tensor, x):
    if isinstance(tensor, torch.Tensor):
        return F.max_pool2d(tensor, x)
    else:
        return reflectTensor(["call", "max_pool2d", [tensor]])

def F_log_softmax(tensor, dim):
    if isinstance(tensor, torch.Tensor):
        return F.log_softmax(tensor, dim)
    else:
        return reflectTensor(["call", "log_softmax", [tensor, dim]])

def __variable(tensor):
    return reflectTensor(["variable", tensor])

def __for_dataloader(src_file, bdfun):
    var_idx = fresh()
    var_data = RepTensor(reflect(freshTensor()).n)
    var_target = fresh()

    def capture(f):
        try: return (False, reify(f, var_idx, var_data, var_target))
        except NonLocalReturnValue as e:
            return e.value

    bodyret, bodyp = capture(bdfun)
    rval = reflectTensor(["for_dataloader", src_file, [var_idx, var_data, var_target], bodyp])
    if not bodyret:
        return rval
    else:
        raise Exception("while: return in body not allowed")

