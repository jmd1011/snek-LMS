from pylms import ast, lms

def test_ast_power():

	@ast
	def power(b, x):
	    if (x == 0): return 1
	    else: return b * power(b, x-1)
	
	assert(power(2,3) == 8)
	#assert(power.code == """(def power (b x) ((if (== x 0) (return 1) (return (* b (call power b (- x 1))))))""")
