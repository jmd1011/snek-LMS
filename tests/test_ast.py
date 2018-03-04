from pylms import ast, lms

@ast
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

def test_power():
    assert(power.original(2,3) == 8)
    assert(power(2,3) == 8)

def test_power_code():    
    assert(power.code == """(def power (b x) ((if (== x 0) (return 1) (return (* b (power b (- x 1)))))))""")


@ast
def ifelse(x):
    model.eval()
    if x == 0:
        print("Hello")
    else:
        print("world!")
    return x

def test_ifelse_code():
    assert(ifelse.code == """(def ifelse (x) ((call model eval) (if (== x 0) (print "Hello") (print "world!")) (return x)))""")

