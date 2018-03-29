# from pylms import lms, lmscompile, stage, ast
from .rep import *
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    'newTensor', 'RepTensor', 'nn_linear', 'nn_conv2d', 'optim_SGD',
    'torch_loader', '__variable', '__for_dataloader',
    'trans_compose', 'trans_to_tensor', 'trans_normalize',
    'F_nll_loss', 'F_relu', 'F_dropout', 'F_max_pool2d', 'F_log_softmax', ]

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
    class RepSGD(object):
        def __init__(self, n):
            self.n = n
        def __repr__(self):
            return str(self.n)

        def zero_grad(self):
            return reflectTensor(["call",self,"zero_grad"])

        def step(self):
            return reflectTensor(["call",self,"step"])

    if isinstance(params, list):
        return optim.SGD(params, lr, momentum)

    tmp = reflect([self,[lr,momentum]])
    return RepSGD(tmp.n)

# def torch_loader(dataset, batch_size, shuffle, **kwargs):
def torch_loader(name, train, download, transforms):
    class RepLoader(object):
        def __init__(self, n):
            self.n = n

        @property
        def dataset(self):
            return reflect(["get",self,"dataset"])

        @dataset.setter
        def setdataset(self,v):
            return reflect(["set",self,"dataset",v])

        @dataset.deleter
        def deldataset(self):
            return reflect(["del",self,"dataset"])

        def __repr__(self):
            return str(self.n)
        # def __len__(self):
        #     return reflectTensor(["call",self,"len"])

    tmp = reflect(["loader", [name, train, download, transforms]])
    return RepLoader(tmp.n)

def trans_compose(ts):
    return reflect(["transform","compose",["{}".format(", ".join([str(t) for t in ts]))]])

def trans_to_tensor():
    return reflect(["transform","toTensor"])

def trans_normalize(*tups):
    return reflect(["transform","normalize", ["{}".format(", ".join([str(i) for j in tups for i in j]))]])

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
    class RepVariable(RepTensor):
        def __init__(self, n):
            self.n = n

        @property
        def grad(self):
            return reflect(["get",self,"grad"])

        @grad.setter
        def grad(self, v):
            return reflect(["set",self,"grad",v])

        @grad.deleter
        def grad(self):
            return reflect(["del",self,"grad"])

    tmp = reflectTensor(["variable", tensor])
    return RepVariable(tmp.n)

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

