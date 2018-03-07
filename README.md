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

	[['val', x0, ['*', in, 1]],
	 ['val', x1, ['*', in, x0]],
	 ['val', x2, ['*', in, x1]],
	 x2]

By executing the recursive calls at generation time,
the function `power` has self-specialized to the
known argument 3.

# Generating C/C++ code

If we want to generate C/C++ code, we provide an other decorator `@stage`.

	@stage
	def power3(x):
	    return power(x, 3)

The decorator is generating a module called `module_power` implemented in C/C++. The decorator also overloads the call to `power3` by a call to a C/C++ function.

The C/C++ code produced is the following:

	/*****************************************/
	#include <stdio.h>
	#include <stdlib.h>
	#include <stdint.h>
	#include "module_power3.h"
	using namespace std;
	int x1(int x2) {
	  int32_t x3 = x2 * x2;
	  int32_t x4 = x2 * x3;
	  return x4;
	}

	int32_t entrypoint(int32_t  x0) {
	  return 0;
	}
	/*******************************************/

# How to Build and Run

Snek-LMS requires a working installation of Swig to load generated C code.

    make init                       # install python dependencies
    make build_compiler             # compile the compiler
    make test                       # run testsuite

    python3 demo.py                 # Run the power3 example shown above
