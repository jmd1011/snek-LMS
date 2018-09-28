from __future__ import print_function
from pylms import *
from pylms.rep import *
from pylms.lantern_staging import *

@lms
def run(in_scores,in_words,in_lefts,in_rights,in_dummy):
	word_embedding_size = 300
	hidden_size = 150
	output_size = 5
	learning_rate = 0.05
	word_embedding_data = newTensor() # check this
	tWi = Tensor.randinit(hidden_size, word_embedding_size, 0.01)
	tbi = Tensor.zeros(hidden_size)
	tWo = Tensor.randinit(hidden_size, word_embedding_size, 0.01)
	tbo = Tensor.zeros(hidden_size)
	tWu = Tensor.randinit(hidden_size, word_embedding_size, 0.01)
	tbu = Tensor.zeros(hidden_size)

	tU0i = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU1i = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tbbi = Tensor.zeros(hidden_size)
	tU00f = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU01f = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU10f = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU11f = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tbbf = Tensor.zeros(hidden_size)
	tU0o = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU1o = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tbbo = Tensor.zeros(hidden_size)
	tU0u = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tU1u = Tensor.randinit(hidden_size, hidden_size, 0.01)
	tbbu = Tensor.zeros(hidden_size)

	tWhy = Tensor.randinit(output_size, hidden_size, 0.01)
	tby = Tensor.zeros(output_size)

	def lossFun(scores, words, lefts, rights, dummy):
		initial_loss = Tensor.zeros(1)
		initial_hidd = Tensor.zeros(hidden_size)
		initial_cell = Tensor.zeros(hidden_size)

		init = rep_tuple(initial_loss, initial_hidd, initial_cell)

		def outputs(i):
			if (i >= 0):
				left = outputs(lefts[i])
				right = outputs(rights[i])

				lossL = left[0]
				hiddenL = left[1]
				cellL = left[2]

				lossR = right[0]
				hiddenR = right[1]
				cellR = right[2]

				tArg = Tensor.zeros(output_size)
				score = scores[i]
				tArg.data[score] = 1

				if lefts[i] < 0:
					word = words[i]
					word_data = word_embedding_data[word]
					embedding_tensor = newTensor(word_data, word_embedding_size)

					i_gate = (tWi.dot(embedding_tensor) + tbi).sigmoid()
					o_gate = (tWo.dot(embedding_tensor) + tbo).sigmoid()
					u_value = (tWu.dot(embedding_tensor) + tbu).tanh()
					cell = i_gate * u_value
					hidden = o_gate * cell.tanh()
					pred1 = (tWhy.dot(hidden) + tby).exp()
					pred2 = pred1 / pred1.sum()
					res = pred2.dot(tArg)
					loss = lossL + lossR - res.log()
					ret = rep_tuple(loss, hidden, cell)
					return ret
				else:
					i_gate1 = (tU0i.dot(hiddenL) + tU1i.dot(hiddenR) + tbbi).sigmoid()
					fl_gate = (tU00f.dot(hiddenL) + tU01f.dot(hiddenR) + tbbf).sigmoid()
					fr_gate = (tU10f.dot(hiddenL) + tU11f.dot(hiddenR) + tbbf).sigmoid()
					o_gate1 = (tU0o.dot(hiddenL) + tU1o.dot(hiddenR) + tbbo).sigmoid()
					u_value1 = (tU0u.dot(hiddenL) + tU1u.dot(hiddenR) + tbbu).tanh()
					cell1 = i_gate1 * u_value1 + fl_gate * cellL + fr_gate * cellR
					hidden1 = o_gate1 * cell1.tanh()
					pred11 = (tWhy.dot(hidden1) + tby).exp()
					pred21 = pred11 / pred11.sum()
					res1 = pred21.dot(tArg)
					loss1 = lossL + lossR - res1.log()
					ret1 = rep_tuple(loss1, hidden1, cell1)
					return ret1
			else:
				return init

		return outputs(0)[0]
	__def_staged(lossFun, in_scores,in_words,in_lefts,in_rights,in_dummy)
	x = __call_staged(lossFun, in_scores,in_words,in_lefts,in_rights,in_dummy)
	return lantern_train(x)

print("==============================================================")
print("=======================ORIGINAL SOURCE========================")
print("==============================================================")
print(run.original_src)

print("==============================================================")
print("========================STAGED SOURCE=========================")
print("==============================================================")
print(run.src)

@stageTensor
def runX(in_scores,in_words,in_lefts,in_rights,in_dummy):
	return run(in_scores,in_words,in_lefts,in_rights,in_dummy)

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
