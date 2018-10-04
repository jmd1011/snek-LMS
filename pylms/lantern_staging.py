from .rep import *

__all__ = [
	"tensor_zeros", "tensor_randinit", "lantern_train"
]

def tensor_zeros(size):
	return reflect(["call", "tensor_zeros", size]);
def tensor_randinit(s1, s2, n):
	return reflect(["call","tensor_randinit",s1,s2,n])
def lantern_train(model):
	return reflect(["call", "lantern_train", model])
