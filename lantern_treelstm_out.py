from __future__ import print_function
from pylms import *
from pylms.rep import *
from pylms.lantern_staging import *

def run(a,b,c,d,e):
    try:
        word_embedding_size = 300
        hidden_size = 150
        output_size = 5
        learning_rate = 0.05
        word_embedding_data = newTensor()
        tWi = tensor_randinit(hidden_size, word_embedding_size, 0.01)
        tbi = tensor_zeros(hidden_size)
        tWo = tensor_randinit(hidden_size, word_embedding_size, 0.01)
        tbo = tensor_zeros(hidden_size)
        tWu = tensor_randinit(hidden_size, word_embedding_size, 0.01)
        tbu = tensor_zeros(hidden_size)
        tU0i = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU1i = tensor_randinit(hidden_size, hidden_size, 0.01)
        tbbi = tensor_zeros(hidden_size)
        tU00f = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU01f = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU10f = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU11f = tensor_randinit(hidden_size, hidden_size, 0.01)
        tbbf = tensor_zeros(hidden_size)
        tU0o = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU1o = tensor_randinit(hidden_size, hidden_size, 0.01)
        tbbo = tensor_zeros(hidden_size)
        tU0u = tensor_randinit(hidden_size, hidden_size, 0.01)
        tU1u = tensor_randinit(hidden_size, hidden_size, 0.01)
        tbbu = tensor_zeros(hidden_size)
        tWhy = tensor_randinit(output_size, hidden_size, 0.01)
        tby = tensor_zeros(output_size)

        def lossFun(scores, words, lefts, rights, dummy):
            try:
                initial_loss = tensor_zeros(1)
                initial_hidd = tensor_zeros(hidden_size)
                initial_cell = tensor_zeros(hidden_size)
                init = rep_tuple(initial_loss, initial_hidd, initial_cell)

                def outputs(i):
                    try:

                        def then_2():
                            left = __call_staged(outputs, lefts[i])
                            right = __call_staged(outputs, rights[i])
                            lossL = left[0]
                            hiddenL = left[1]
                            cellL = left[2]
                            lossR = right[0]
                            hiddenR = right[1]
                            cellR = right[2]
                            tArg = tensor_zeros(output_size)
                            score = scores[i]
                            tArg.data_set(score, 1)

                            def then_1():
                                try:
                                    word = words[i]
                                    word_data = word_embedding_data[word]
                                    embedding_tensor = newTensor(word_data, word_embedding_size)
                                    i_gate = (tWi.dot(embedding_tensor) + tbi).sigmoid()
                                    o_gate = (tWo.dot(embedding_tensor) + tbo).sigmoid()
                                    u_value = (tWu.dot(embedding_tensor) + tbu).tanh()
                                    cell = (i_gate * u_value)
                                    hidden = (o_gate * cell.tanh())
                                    pred1 = (tWhy.dot(hidden) + tby).exp()
                                    pred2 = (pred1 / pred1.sum())
                                    res = pred2.dot(tArg)
                                    loss = ((lossL + lossR) - res.log())
                                    ret = rep_tuple(loss, hidden, cell)
                                    __return(ret)
                                except NonLocalReturnValue as r:
                                    return r.value

                            def else_1():
                                try:
                                    i_gate1 = ((tU0i.dot(hiddenL) + tU1i.dot(hiddenR)) + tbbi).sigmoid()
                                    fl_gate = ((tU00f.dot(hiddenL) + tU01f.dot(hiddenR)) + tbbf).sigmoid()
                                    fr_gate = ((tU10f.dot(hiddenL) + tU11f.dot(hiddenR)) + tbbf).sigmoid()
                                    o_gate1 = ((tU0o.dot(hiddenL) + tU1o.dot(hiddenR)) + tbbo).sigmoid()
                                    u_value1 = ((tU0u.dot(hiddenL) + tU1u.dot(hiddenR)) + tbbu).tanh()
                                    cell1 = (((i_gate1 * u_value1) + (fl_gate * cellL)) + (fr_gate * cellR))
                                    hidden1 = (o_gate1 * cell1.tanh())
                                    pred11 = (tWhy.dot(hidden1) + tby).exp()
                                    pred21 = (pred11 / pred11.sum())
                                    res1 = pred21.dot(tArg)
                                    loss1 = ((lossL + lossR) - res1.log())
                                    ret1 = rep_tuple(loss1, hidden1, cell1)
                                    __return(ret1)
                                except NonLocalReturnValue as r:
                                    return r.value
                            __return(__if((lefts[i] < 0), then_1, else_1))

                        def else_2():
                            __return(init)
                        __if((i >= 0), then_2, else_2)
                    except NonLocalReturnValue as r:
                        return r.value
                ri = Rep('i')
                __def_staged(outputs, ri)
                __return(__call_staged(outputs, 0)[0])
            except NonLocalReturnValue as r:
                return r.value
        __def_staged(lossFun,a,b,c,d,e)
        __return(__call_staged(lossFun, a,b,c,d,e))
    except NonLocalReturnValue as r:
        return r.value

@stageTensor
def runX(a,b,c,d,e):
    return run(a,b,c,d,e)

print(runX.code)