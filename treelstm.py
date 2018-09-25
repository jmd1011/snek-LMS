from __future__ import print_function
from pylms import lms, stage, stageTensor
from pylms.rep import Rep

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

@lms
def run(dummy):
	import torch
	import torch.nn as nn
	import torch.nn.functional as F

	# implementation pulled from https://github.com/dasguptar/treelstm.pytorch
	# tree object from stanfordnlp/treelstm
	class Tree(object):
	    def __init__(self):
	        self.parent = None
	        self.num_children = 0
	        self.children = list()

	    def add_child(self, child):
	        child.parent = self
	        self.num_children += 1
	        self.children.append(child)

	    def size(self):
	        if getattr(self, '_size'):
	            return self._size
	        count = 1
	        for i in range(self.num_children):
	            count += self.children[i].size()
	        self._size = count
	        return self._size

	    def depth(self):
	        if getattr(self, '_depth'):
	            return self._depth
	        count = 0
	        if self.num_children > 0:
	            for i in range(self.num_children):
	                child_depth = self.children[i].depth()
	                if child_depth > count:
	                    count = child_depth
	            count += 1
	        self._depth = count
	        return self._depth

    # module for childsumtreelstm
	class ChildSumTreeLSTM(nn.Module):
	    def __init__(self, in_dim, mem_dim):
	        super(ChildSumTreeLSTM, self).__init__()
	        self.in_dim = in_dim
	        self.mem_dim = mem_dim
	        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
	        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
	        self.fx = nn.Linear(self.in_dim, self.mem_dim)
	        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

	    def node_forward(self, inputs, child_c, child_h):
	        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

	        iou = self.ioux(inputs) + self.iouh(child_h_sum)
	        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
	        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

	        z = self.fx(inputs)
	        f = F.sigmoid(
	            self.fh(child_h) +
	            z.repeat(len(child_h), 1)
	        )
	        fc = torch.mul(f, child_c)

	        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
	        h = torch.mul(o, F.tanh(c))
	        return c, h

	    def forward(self, tree, inputs):
	        for idx in range(tree.num_children):
	            self.forward(tree.children[idx], inputs)

	        if tree.num_children == 0:
	            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
	            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
	        else:
	            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
	            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

	        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
	        return tree.state

	# module for distance-angle similarity
	class Similarity(nn.Module):
	    def __init__(self, mem_dim, hidden_dim, num_classes):
	        super(Similarity, self).__init__()
	        self.mem_dim = mem_dim
	        self.hidden_dim = hidden_dim
	        self.num_classes = num_classes
	        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
	        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

	    def forward(self, lvec, rvec):
	        mult_dist = torch.mul(lvec, rvec)
	        abs_dist = torch.abs(torch.add(lvec, -rvec))
	        vec_dist = torch.cat((mult_dist, abs_dist), 1)

	        out = F.sigmoid(self.wh(vec_dist))
	        out = F.log_softmax(self.wp(out), dim=1)
	        return out


	# putting the whole model together
	class SimilarityTreeLSTM(nn.Module):
	    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
	        super(SimilarityTreeLSTM, self).__init__()
	        self.emb = nn.Embedding(vocab_size, in_dim, PAD, sparse=sparsity)
	        if freeze:
	            self.emb.weight.requires_grad = False
	        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
	        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

	    def forward(self, ltree, linputs, rtree, rinputs):
	        linputs = self.emb(linputs)
	        rinputs = self.emb(rinputs)
	        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
	        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
	        output = self.similarity(lstate, rstate)
	        return output

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
