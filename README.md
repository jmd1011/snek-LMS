# snek-LMS
A prototype implementation of [LMS](https://scala-lms.github.io) in Python.

[![Build Status](https://travis-ci.org/jmd1011/snek-LMS.svg?branch=master)](https://travis-ci.org/jmd1011/snek-LMS)

# Lightweight Syntax for Building Computation Graphs

Snek-LMS provides a decorator `@lms` that extends Python's operator overloading capabilities to many built-in operations. This function:

	@lms
	def loop(n):
	    x = 0
	    while x < n:
	        x = x + 1
	    return x

Will be converted (roughly) to:

	def loop(n):
        x = __new_var()
        __assign(x, 0)

        def cond$1():
            return (__read(x) < n)
        
        def body$1():
            __assign(x, (__read(x) + 1))
        
        __while(cond$1, body$1)

        return __read(x)

Function like `__while` are overloaded to construct graph
nodes for each operation. Hence, *executing* this transformed 
version will *build* an IR (i.e., a computation graph) that 
represents its computation:

	[['val', x5, ['new_var']],
	 ['val', x6, ['assign', x5, 0]],
	 ['val', x7, ['while',
	    [['val', x7, ['read', x5]],
	     ['val', x8, ['<', x7, in]],
	     x8],
	    [['val', x7, ['read', x5]],
	     ['val', x8, ['+', x7, 1]],
	     ['val', x9, ['assign', x5, x8]],
	     None]]],
	 ['val', x8, ['read', x5]], x8]

From here, we can translate further: either directly to C code
or to a system like TensorFlow.


# Multi-Stage Programming

The power of LMS over systems like Cython and Numba that work
directly on Python syntax comes through interleaving
generation-time computation with IR construction.

For example:

	@lms
	def power(b, x):
	    if (x == 0): return 1
	    else: return b * power(b, x-1)

Invoking `lmscompile(lambda x: power(x,3))` produces the
following IR:

	[['val', x0, ['*', in, 1]], ['val', x1, ['*', in, x0]], ['val', x2, ['*', in, x1]], x2]

By executing the recursive calls at generation time,
the function `power` has self-specialized to the 
known arguments.



# How to Build and Run

Snek-LMS requires a working installation of Swig to load generated C code.

    make build_compiler             # compile the compiler
    java -jars compiler/target/scala-2.11/sneklms.jar &    # start the server (maybe better to do it in an other shell rather than in background)
    make test                       # run testsuite
    python3 compiler/pipeline.py    # simple python script calling the compiler
