# snek-LMS
A prototype implementation of [LMS](https://scala-lms.github.io) in Python.

[![Build Status](https://travis-ci.org/jmd1011/snek-LMS.svg?branch=master)](https://travis-ci.org/jmd1011/snek-LMS)

# Lantern Demonstration

To demonstrate the full power of Snek-LMS, we've teamed up with the folks over at [Lantern](https://feiwang3311.github.io/Lantern/)! In this demo, we take some (nearly) ordinary PyTorch code which runs the canonical machine learning example of MNIST. Note that in the interest of brevity, we elect to simplify our model to have two linear layers, rather than the conventional convolutional layers.

# Running the Demo

Here we are, the exciting part!

## Prerequisites

As always, the first step is to ensure all necessary prerequisites are installed. We detail the necessary tools and how to install them below.

### python3

Snek-LMS requires at least Python 3.5.

[Installing Python 3.5 for Linux](http://docs.python-guide.org/en/latest/starting/install3/linux/)

[Installing Python 3.5 for OSX](http://docs.python-guide.org/en/latest/starting/install3/osx/)

[Installing Python 3.5 for Windows](http://docs.python-guide.org/en/latest/starting/install3/win/)

### pip3

[Installing pip3](https://pip.pypa.io/en/stable/installing/)

<!-- Installing pip3 for Linux: `sudo apt install pip3` or equivalent

[Install pip3 for OSX](http://itsevans.com/install-pip-osx/)

 -->

### swig/swig3.0

We have found that OSX users require the use of SWIG, whereas other users have reported SWIG 3.0 working best for them. Be sure to select the correct version for your system!

[Installing SWIG](http://www.swig.org/Doc3.0/Preface.html)

### PyTorch

PyTorch has an easy to use installation guide on their site, linked below.

[Installing PyTorch](http://pytorch.org/)

### g++

Installing g++ for Linux: `sudo apt install g++` (or equivalent)

[Installing g++ for OSX](http://www-scf.usc.edu/~csci104/20142/installation/gccmac.html)

[Installing g++ for Windows](http://www1.cmc.edu/pages/faculty/alee/g++/g++.html)

### Other Requirements

With these in place, you should be able to perform `make init` and have all other requirements automatically installed.

## Punch it, Chewy!

With everything installed, perform the following steps to actually get things moving!

(If you skipped to this section, don't forget to run `make init` to get all prerequisites installed.)

You should only need to run these once to set up Snek-LMS:

- `make data #this downloads and sets up the MNIST data`
- `make build_compiler`

Finally, we can run the demo:

- `python3 lantern_demo.py`

This will give a giant wall of text, separated into 5 categories:

1. ORIGINAL SOURCE
	i. The PyTorch code which we're transforming.
2. STAGED SOURCE
	i. The transformed PyTorch code.
3. IR CODE
	i. The [S-Expr](https://en.wikipedia.org/wiki/S-expression) intermediate representation which will be read by our Scala code (in the `compiler` directory) and used to generate Lantern code.
4. GENERATED CODE
	i. The C++ code output by Lantern.

The generated code is also available for inspection in the `gen` folder (in our case, it's the `module_runX.cpp` file).

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

	[['let', x5, ['new_var']],
	 ['let', x6, ['assign', x5, 0]],
	 ['let', x7, ['while',
	    [['let', x7, ['read', x5]],
	     ['let', x8, ['<', x7, in]],
	     x8],
	    [['let', x7, ['read', x5]],
	     ['let', x8, ['+', x7, 1]],
	     ['let', x9, ['assign', x5, x8]],
	     None]]],
	 ['let', x8, ['read', x5]], x8]

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

	[['let', x0, ['*', in, 1]],
	 ['let', x1, ['*', in, x0]],
	 ['let', x2, ['*', in, x1]],
	 x2]

By executing the recursive calls at generation time,
the function `power` has self-specialized to the
known argument 3.

# Generating C/C++ code

If one wants to generate C/C++ code, we provide another decorator `@stage`.

	@stage
	def power3(x):
	    return power(x, 3)

The decorator generates a module called `module_power3` implemented in C/C++. The decorator also overloads the call to `power3` by a call to a C/C++ function.

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

    make init                       # install python dependencies (may need sudo)
    make build_compiler             # compile the compiler
    make test                       # run testsuite

    python3 demo.py                 # Run the power3 example shown above
