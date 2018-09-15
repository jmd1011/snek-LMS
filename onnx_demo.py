from __future__ import print_function
import onnx

onnx_model = onnx.load('/home/james/Research/onnx_tutorials/tutorials/assets/squeezenet.onnx')

print('Model: {0}'.format(onnx_model))