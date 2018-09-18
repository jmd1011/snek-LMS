from __future__ import print_function
import onnx
from pylms.nn_staging import *
from pylms import lms, stageTensor

@lms
def run(dummy):
	import torch
	# from torch.autograd import Variable
	# set_backend('lantern')
	onnx_model = onnx.load('/home/fei/bitbucket/snek-LMS/model.onnx')
	input_file = "data.csv"
	x = onnx.run(onnx_model, input_file)
	print(x)
	# x = Variable(torch.randn(1, 3, 224, 224), True)
	# ...
	# (let (x0 (onnx_load test.onnx)) (let (x1 (...)) (let (x2 (onnx_run (x0 data.csv))))))

# @stageTensor
# def runX(x):
# 	return run(x)

@stageTensor
def runX(x):
	return run(x)

print(runX.code)
runX(0)