from __future__ import print_function
import onnx
from pylms.nn_staging import *
from pylms import lms, stageTensor

@lms
def run(dummy):
	import torch
	# from torch.autograd import Variable
	# set_backend('lantern')
	onnx_model = onnx.load('/home/james/Research/onnx_tutorials/tutorials/assets/squeezenet.onnx')
	input_data = 3
	x = onnx.run(onnx_model, input_data)
	print(x)
	# x = Variable(torch.randn(1, 3, 224, 224), True)
	# ...
	# (let (x0 (onnx_load test.onnx)) (let (x1 (...)) (let (x2 (onnx_run (x0 x1))))))

# @stageTensor
# def runX(x):
# 	return run(x)

@stageTensor
def runX(x):
	return run(x)

print(runX.code)
# run('a')