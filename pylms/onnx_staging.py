from .rep import *

__all__ = ['onnx_load', 'onnx_run']

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

def onnx_load(filename):
    return reflectTensor(["onnx_load", filename])

def onnx_run(model, data):
    return reflectTensor(["onnx_run", [model, data]])