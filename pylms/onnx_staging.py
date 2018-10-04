from .rep import *

__all__ = [
    'onnx_load', 'lantern_run', 'lantern_train'
]

def onnx_load(filename):
    return reflect(["onnx_load", filename])

def lantern_run(model, data):
    return reflect(["lantern_run", [model, data]])

def lantern_train(model, data):
    return reflect(["lantern_train", [model, data]])
