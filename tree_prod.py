from __future__ import print_function
from pylms import lms, stage
from pylms.rep import Rep, reflect

import torch

def reflectTree(args):
	rep = reflect(args)
	return Tree(rep.n)

class Tree(Rep):
	def __init__(self, n):
		super().__init__(n)

	@property
	def is_empty(self):
		return reflectTree(["getattr",self,"is_empty"])

	@property
	def left(self):
		return reflectTree(["getattr",self,"left"])

	@property
	def right(self):
		return reflectTree(["getattr",self,"right"])

	def insert(self, value):
		return reflectTree(["call",self,"insert",value])

@lms
def run(x):
	def tree_prod(base, tree):
		if tree.is_empty:
			return base
		else:
			l = tree_prod(base, tree.left)
			r = tree_prod(base, tree.right)
			return l * r * tree.value

	return tree_prod(1, x)

print("==============================================================")
print("=======================ORIGINAL SOURCE========================")
print("==============================================================")
print(run.original_src)

print("==============================================================")
print("========================STAGED SOURCE=========================")
print("==============================================================")
print(run.src)

@stage
def runX(x):
    return run(Tree(x))

print("==============================================================")
print("===========================IR CODE============================")
print("==============================================================")
print(runX.code)

print("==============================================================")
print("========================GENERATED CODE========================")
print("==============================================================")
print(runX.Ccode)

print("==============================================================")
print("========================EXECUTING CODE========================")
print("==============================================================")
runX(5)

print("==============================================================")
print("========================EXITING PROGRAM=======================")
print("==============================================================")
