from .rep import *

__all__ = [
	"tensor_zeros", "tensor_randinit", "lantern_train"
]

def tensor_zeros(size):
	return reflectTensor(["call", "tensor_zeros", size]);
def tensor_randinit(s1, s2, n):
	return reflectTensor(["call","tensor_randinit",s1,s2,n])
def lantern_train(model):
	return reflectTensor(["call", "lantern_train", model])