from __future__ import print_function
import onnx
import os
from pylms.nn_staging import *
from pylms import lms, stageTensor

onnx_modeldir = os.environ['HOME'] + "/onnx_models"

@lms
def run(dummy):
	# import torch
	# from torch.autograd import Variable
	onnx_model = onnx.load('{}/squeezenet/model.onnx'.format(onnx_modeldir))
	# x = Variable(torch.randn(1, 3, 224, 224), True)
	input_file = 'test.csv'
	res = lantern.run(onnx_model, input_file) # .data.numpy())
	print(res)

print("==============================================================")
print("=======================ORIGINAL SOURCE========================")
print("==============================================================")
print(run.original_src)

print("==============================================================")
print("========================STAGED SOURCE=========================")
print("==============================================================")
print(run.src)

@stageTensor
def runX(x):
	return run(x)

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
runX(0)

print("==============================================================")
print("========================EXITING PROGRAM=======================")
print("==============================================================")
