from .rep import *

__all__ = [
	"tensor_zeros", "tensor_randinit"
]

def tensor_zeros(size):
	return reflectTensor(["call", "tensor_zeros", size]);
def tensor_randinit(s1, s2, n):
	return reflectTensor(["call","tensor_randinit",s1,s2,n])