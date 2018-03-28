# from pylms import lms, lmscompile, stage, ast
from .rep import *
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['nn_linear', 'nn_conv2d', 'RepTensor', 'optim_SGD', 'F_nll_loss', '__variable', 'rep_train_loader_tensor', 'rep_train_loader_fresh', '__for_dataloader', 'F_relu', 'F_dropout', 'F_max_pool2d', 'F_log_softmax']

stFresh = 0

def freshTensor(*dims):
    global stFresh
    stFresh += 1
    return RepTensor("t"+str(stFresh-1), *dims)

class RepTensor(Rep):
    def __init__(self, n, *dims):
        super().__init__(n)
        self.dims = [i for i in dims]
    def __mul__(self, m):
        return reflect(["dot",self,m])
    def __repr__(self):
        return "[tensor, [{}]]".format(", ".join(list(map(str, self.dims))))

    def data_get(self, i):
        return reflect(["array-get", self.n, "data", i])

    def backward(self):
        return reflect(["call", self.n, "backward"])

    def conv2d(self, kernel):
        rep = reflect(["call", self.n, "conv2d", kernel.n])
        return RepTensor(self.n, None)

    def view(self, *dims):
        rep = reflect(["call", self.n, "view", dims])
        return(RepTensor(self.n, dims))

def rep_train_loader_tensor():
    rept = reflect(freshTensor(None))
    return rept

def rep_train_loader_fresh():
    return fresh()


def nn_linear(hlsize, outsize):
    class Linear(object):
        def __init__(self):
            self.weight = reflect(freshTensor(outsize, hlsize))
            self.bias = reflect(freshTensor(outsize))
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
            tmp = reflect(freshTensor(outSize, inSize, kernel_size, kernel_size))
            self.kernel = RepTensor(tmp.n, outSize, inSize, kernel_size, kernel_size)
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
                tmp = reflect(["call", self, "zero_grad"])
                return RepTensor(tmp.n, None)
            else:
                return self.optim.zero_grad()

        def step(self):
            if self.staged:
                tmp = reflect(["call", self, "step"])
            else:
                return self.optim.step()

    return RepSGD("SGD")

def F_nll_loss(output, target, size_average=True):
    if isinstance(output, Variable):
        return F.nll_loss(output, target, size_average)
    else:
        tmp = reflect(["call", "nll_loss", [output, target, size_average]])
        return RepTensor(tmp.n, None)

def F_relu(tensor):
    if isinstance(tensor, torch.Tensor):
        return F.relu(tensor)
    else:
        tmp = reflect(["call", "relu", [tensor]])
        return RepTensor(tmp.n, None)

def F_dropout(tensor, training):
    if isinstance(tensor, torch.Tensor):
        return F.dropout(tensor, training=training)
    else:
        tmp = reflect(["call", "dropout", [tensor, training]])
        return RepTensor(tmp.n, None)

def F_max_pool2d(tensor, x):
    if isinstance(tensor, torch.Tensor):
        return F.max_pool2d(tensor, x)
    else:
        tmp = reflect(["call", "max_pool2d", [tensor]])
        return RepTensor(tmp.n, None)

def F_log_softmax(tensor, dim):
    if isinstance(tensor, torch.Tensor):
        return F.log_softmax(tensor, dim)
    else:
        tmp = reflect(["call", "log_softmax", [tensor, dim]])
        return RepTensor(tmp.n, None)

def __variable(tensor):
    return tensor

def __for_dataloader(src_file, bdfun):
    var_idx = fresh()
    var_data = reflect(freshTensor())
    var_target = fresh()

    def capture(f):
        try: return (False, reify(f, var_idx, var_data, var_target))
        except NonLocalReturnValue as e:
            return e.value

    bodyret, bodyp = capture(bdfun)
    rval = reflect(["for_dataloader", src_file, [var_idx, var_data, var_target], bodyp])
    if not bodyret:
        return rval
    else:
        raise Exception("while: return in body not allowed")

