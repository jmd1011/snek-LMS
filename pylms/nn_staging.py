# from pylms import lms, lmscompile, stage, ast
from .rep import *
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    'torch_loader', 'torch_abs', 'torch_add', 'torch_mul', 'torch_sum',
    'nn_conv2d', 'nn_linear',
    'F_dropout', 'F_log_softmax', 'F_max_pool2d',
    'F_nll_loss', 'F_relu', 'F_sigmoid', 'F_tanh',
    'trans_compose', 'trans_to_tensor', 'trans_normalize',
    'optim_SGD',
    'rep_variable', '_for_dataloader',
    ]


#############################################################
####################### torch Methods #######################
#############################################################

def torch_loader(name, train, download, transforms):
    class RepLoader(object):
        def __init__(self, n):
            self.n = n

        @property
        def dataset(self):
            return reflect(["getattr",self,"dataset"])

        @dataset.setter
        def setdataset(self,v):
            return reflect(["setattr",self,"dataset",v])

        def __len__(self):
            return len(self.n)

        def __repr__(self):
            return str(self.n)

    tmp = reflect(["loader", [name, train, download, transforms]])
    return RepLoader(tmp.n)

def torch_abs(t1):
    if not isinstance(t1, Rep):
        return torch.abs(t1)
    return reflect(["call", "abs", [t1]])

def torch_add(t1, t2):
    if not isinstance(t1, Rep) and not isinstance(t2, Rep):
        return torch.add(t1, t2)
    return reflect(["call", "add", [t1, t2]])

def torch_cat(t1, dim):
    if not isinstance(t1, Rep) and not isinstance(dim, Rep):
        return torch.cat(t1, dim)
    return reflect(["call", "cat", [t1, dim]])

def torch_mul(t1, t2):
    if not isinstance(t1, Rep) and not isinstance(t2, Rep):
        return torch.mul(t1, t2)
    return reflect(["call", "mul", [t1, t2]])

def torch_split(iou, size, dim):
    if not isinstance(t1, Rep)\
      and not isinstance(size, Rep)\
      and not isinstance(dim, Rep):
        return torch.split(t1, t2)
    return reflect(["call", "split", [iou, size, dim]])

def torch_sum(t1, t2):
    if not isinstance(t1, Rep) and not isinstance(t2, Rep):
        return torch.sum(t1, t2)
    return reflect(["call", "sum", [t1, t2]])


#############################################################
###################### torch.nn Methods #####################
#############################################################

def newTensor(*dims):
    return reflect(['call','tensor',['{}'.format(', '.join(list(map(str, dims))))]])

def nn_linear(hlsize, outsize):
    class Linear(object):
        def __init__(self):
            self.rep = reflect(["call","nn_linear",[outsize,hlsize]])
            self.linear = None

        def __call__(self, tensor):
            if isinstance(tensor, torch.Tensor): #unstaged
                if self.linear is None:
                    self.linear = nn.Linear(hlsize, outsize)

                return self.linear(tensor)
            else: #staged
                return self.rep(tensor)
        def __repr__(self):
            return str(self.rep)
    return Linear()

def nn_conv2d(outSize, inSize, kernel_size, bias):
    class Conv2d(object):
        def __init__(self):
            self.kernel = Rep(fresh(), outSize, inSize, kernel_size, kernel_size) # newTensor(outSize, inSize, kernel_size, kernel_size)
            self.conv2d = None

        def __call__(self, tensor):
            if isinstance(tensor, torch.Tensor): #unstaged
                if self.conv2d is None:
                    self.conv2d = nn.Conv2d(hlsize, outsize, kernel_size=kernel_size, bias=bias)
                return self.conv2d(tensor)
            else: #staged
                return tensor.conv2d(self.kernel)
    return Conv2d()

#############################################################
################## torch.transforms Methods #################
#############################################################

def trans_compose(ts):
    return reflect(["transform","compose",
        ["{}".format(", ".join([str(t) for t in ts]))]])

def trans_to_tensor():
    return reflect(["transform","toTensor"])

def trans_normalize(*tups):
    return reflect(["transform","normalize",
        ["{}".format(", ".join([str(i) for j in tups for i in j]))]])


##############################################################
################ torch.nn.functional Methods #################
##############################################################

def F_dropout(tensor):
    if not isinstance(tensor, Rep):
        return F.dropout(tensor)
    else:
        return reflect(["call", "dropout", [tensor]])

def F_log_softmax(tensor, dim):
    if not isinstance(tensor, Rep):
        return F.log_softmax(tensor, dim)
    else:
        return reflect(["call", "log_softmax", [tensor, 'dim={}'.format(dim)]])

def F_max_pool2d(tensor, x):
    if not isinstance(tensor, Rep):
        return F.max_pool2d(tensor, x)
    else:
        return reflect(["call", "max_pool2d", [tensor]])

def F_nll_loss(output, target, size_average=True):
    if not isinstance(output, Rep):
        return F.nll_loss(output, target, size_average)
    else:
        return reflect(["call", "nll_loss", [output, target, size_average]])

def F_relu(tensor):
    if not isinstance(tensor, Rep):
        return F.relu(tensor)
    else:
        return reflect(["call", "relu", [tensor]])

def F_sigmoid(t1, t2):
    if not isinstance(t1, Rep) and not isinstance(t2, Rep):
        return F.sigmoid(t1, t2)
    else:
        return reflect(["call", "sigmoid", [t1, t2]])

def F_tanh(t):
    if not isinstance(tensor, Rep):
        return F.tanh(tensor)
    else:
        return reflect(["call", "tanh", [tensor]])

###################################################
################## Miscellaneous ##################
###################################################

def optim_SGD(params, lr, momentum):
    class RepSGD(object):
        def __init__(self, n):
            self.n = n
        def __repr__(self):
            return str(self.n)

        def zero_grad(self):
            return reflect(["call",self,"zero_grad"])

        def step(self):
            return reflect(["call",self,"step"])

    if isinstance(params, list):
        if isinstance(params[0], torch.Tensor):
            return optim.SGD(params, lr, momentum)

    tmp = reflect(['call',"SGD",[params, lr,momentum]])
    return RepSGD(tmp.n)

def rep_variable(tensor, volatile=False):
    if not isinstance(tensor, Rep):
        return torch.Variable(tensor, volatile=volatile)
    class RepVariable(Rep):
        def __init__(self, n):
            self.n = n

        @property
        def grad(self):
            return reflect(["getattr",self,"grad"])

        @grad.setter
        def grad(self, v):
            return reflect(["setattr",self,"grad",v])

    tmp = reflect(["call", "variable", [tensor, volatile]])
    return RepVariable(tmp.n)

def _for_dataloader(src_file, bdfun):
    var_idx = fresh()
    var_data = fresh() # freshTensor()
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
