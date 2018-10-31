from .rep import *

__all__ = [
	"tensor_zeros", "tensor_randinit", "lantern_train",'lantern_read'
]

def tensor_zeros(size):
	return reflect(["call", "tensor_zeros", size]);
def tensor_randinit(s1, s2, n):
	return reflect(["call","tensor_randinit",s1,s2,n])
def lantern_train(model,*args):
	return reflect(["call", "lantern_train", model,args])
def lantern_read(path):
	return reflect(['call','lantern_read',path])