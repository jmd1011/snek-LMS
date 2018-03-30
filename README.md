# snek-LMS
A prototype implementation of [LMS](https://scala-lms.github.io) in Python.

[![Build Status](https://travis-ci.org/jmd1011/snek-LMS.svg?branch=master)](https://travis-ci.org/jmd1011/snek-LMS)

# Lantern Demonstration

To demonstrate the full power of Snek-LMS, we've teamed up with the folks over at [Lantern](https://feiwang3311.github.io/Lantern/)! In this demo, we take some (nearly) ordinary PyTorch code which runs the canonical machine learning example of MNIST. Note that in the interest of brevity, we elect to simplify our model to have two linear layers, rather than the conventional convolutional layers. Even in this simple case, we see a 3x speedup by generating native C++ code!

# Running the Demo

Here we are, the exciting part! Be sure to get all of the [prerequisites](https://github.com/jmd1011/snek-LMS/tree/demo#prerequisites) installed, and let's dive right in!

## Punch it, Chewy!

You should only need to run these once to set up Snek-LMS:

- `make init`
- `make data #this downloads and sets up the MNIST data`
- `make build_compiler`

With these in place, we can start training!

### Training in PyTorch

Let's take a look at some of the PyTorch code we'll be working with (available in `pytorch_demo.py`):

```
def run():
    ...
    train_loader = torch.utils.data.DataLoader(...)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net()
    optimizer = optim.SGD(...)

    def train(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            ...
            loss.backward()
            optimizer.step()
            if (((batch_idx + 1) % args.log_interval) == 0):
                #print_time_and_loss
                ...

    for epoch in range(1, args.epochs + 1):
        train(epoch)
```

As shown, this handles training our model and calculating the training loss.

### Running PyTorch

Running this code using `time python3 pytorch_demo.py` yields something similar to the following output:

```

Train Epoch: 1 [6000/60000 (10%)]   Loss: 0.971245
Train Epoch: 1 [12000/60000 (20%)]  Loss: 0.702314
Train Epoch: 1 [18000/60000 (30%)]  Loss: 0.603477
Train Epoch: 1 [24000/60000 (40%)]  Loss: 0.530881
Train Epoch: 1 [30000/60000 (50%)]  Loss: 0.487666
Train Epoch: 1 [36000/60000 (60%)]  Loss: 0.456104
Train Epoch: 1 [42000/60000 (70%)]  Loss: 0.431443
Train Epoch: 1 [48000/60000 (80%)]  Loss: 0.412651
Train Epoch: 1 [54000/60000 (90%)]  Loss: 0.396839
Train Epoch: 1 [60000/60000 (100%)] Loss: 0.376887
Train Epoch: 2 [6000/60000 (10%)]   Loss: 0.218387
Train Epoch: 2 [12000/60000 (20%)]  Loss: 0.222979
...
Train Epoch: 10 [54000/60000 (90%)]   Loss: 0.074284
Train Epoch: 10 [60000/60000 (100%)]    Loss: 0.072710

real    3m25.236s
user    3m21.780s
sys     0m9.404s
```

While a training loss of only 0.07 is great for 10 epochs, the fact that this can take upwards of 5 minutes definitely isn't so great.

### Training in Snek-LMS

Let's see if we can do better with Snek-LMS and Lantern!

We perform some very simple modifications to our training function and add some bootstrapping, as follows:

```
from pylms import lms, stage, stageTensor  # add our snek-lms module
from pylms.rep import Rep                  # add our snek-lms module

@lms                                       # add annotation for snek-lms
def run(dummy):
    ...
    train_loader = torch.utils.data.DataLoader(...)

    fc1 = nn.Linear(784, 50)
    fc2 = nn.Linear(50, 10)

    def forward(x):
        x1 = x.view(-1, 784)
        x2 = F.relu(fc1(x1))
        x3 = fc2(x2)
        return F.log_softmax(x3, dim=1)

    optimizer = optim.SGD(...)

    def train(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            ...
            loss.backward()
            optimizer.step()
            if (((batch_idx + 1) % args.log_interval) == 0):
                #print_time_and_loss
                ...

    idx = 0
    while idx < args.epochs:
        idx = idx + 1
        train(idx)

@stage                                     # add annotation and bootstrapping
def runX(x):
    return run(x)

                                           # add pretty printing

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
```

### Running Lantern

Running `time python3 lantern_demo.py` yields a giant wall of text, separated into 5 categories (we elide some for simplicity of presentation):

1. ORIGINAL SOURCE
    1. The PyTorch code which we're transforming. We elide this in our output, as it is visible above.
2. STAGED SOURCE
    1. The transformed PyTorch code.

```
==============================================================
========================STAGED SOURCE=========================
==============================================================
def run(dummy):
    try:
        idx = __var()
        ... #elided for presentation
        train_loader = torch_loader(...)
        fc1 = nn_linear(784, 50)
        fc2 = nn_linear(50, 10)
        optimizer = optim_SGD(...)

        def forward(x):
            try:
                x1 = x.view((- 1), 784)
                x2 = F_relu(fc1(x1))
                x3 = fc2(x2)
                __return(F_log_softmax(x3, dim=1))
            except NonLocalReturnValue as r:
                return r.value

        def train(epoch):
            try:

                def forfunc$1(batch_idx, data, target):
                    ...
                    loss = res.backward()
                    optimizer.step()

                    def then$1():
                        __printf(...)
                    def else$1():
                        pass
                    __if((((batch_idx + 1) % args.log_interval) == 0), then$1, else$1)

                __for_dataloader(train_loader, forfunc$1)

            except NonLocalReturnValue as r:
                return r.value

        __assign(idx, 0)

        def cond$1():
            return (__read(idx) < args.epochs)

        def body$1():
            __assign(idx, (__read(idx) + 1))
            train(__read(idx))
        __while(cond$1, body$1)
    except NonLocalReturnValue as r:
        return r.value
```

3. IR CODE
    1. The [S-Expr](https://en.wikipedia.org/wiki/S-expression) intermediate representation which will be read by our Scala code (in the `compiler` directory) and used to generate Lantern code.

```
==============================================================
===========================IR CODE============================
==============================================================
(def runX (in) (begin (begin (let x0 new (let x1 (transform toTensor) (let x2 (transform normalize (0.1307 0.3081)) (let x3 (transform compose (x1 x2)) (let x4 (loader (MNIST True True x3)) (let x5 (tensor (50 784)) (let x6 (variable x5 False) (let x7 (tensor (50)) (let x8 (variable x7 False) (let x9 (tensor (10 50)) (let x10 (variable x9 False) (let x11 (tensor (10)) (let x12 (variable x11 False) (let x13 (SGD (0.0005 0.0)) (let x14 (set x0 0) (let x15 (print "Start Training") (let x16 (while (begin (let x16 (get x0) (let x17 (< x16 10) x17))) (begin (let x16 (get x0) (let x17 (+ x16 1) (let x18 (set x0 x17) (let x19 (get x0) (let x20 (printf ("Epoch {:.0f}" x19)) (let x21 (get x0) (let x22 new (let x23 (set x22 0.0) (let x26 (for_dataloader x4 (x24 t0 x25) (begin (let x26 (variable t0 True) (let x27 (variable x25 False) (let x28 (call x13 zero_grad) (let x29 (call x26 view (-1 784)) (let x30 (dot x6 x29) (let x31 (+ x30 x8) (let x32 (call relu (x31)) (let x33 (dot x10 x32) (let x34 (+ x33 x12) (let x35 (call log_softmax (x34 1)) (let x36 (call nll_loss (x35 x27 True)) (let x37 (call x36 backward) (let x38 (get x22) (let x39 (array-get x37 data 0) (let x40 (+ x38 x39) (let x41 (set x22 x40) (let x42 (call x13 step) (let x43 (get x22) (let x44 (+ x24 1) (let x45 (% x44 6000) (let x46 (== x45 0) (let x47 (if x46 (begin (let x47 (+ x24 1) (let x48 (len x4) (let x49 (* x24 100.0) (let x50 (len x4) (let x51 (/ x49 x50) (let x52 (/ x43 x24) (let x53 (printf ("Train Epoch: {:.0f} ({}/{} ({:.0f}%))\tLoss: {:.6f}" x21 x47 x48 x51 x52)) None)))))))) (begin None)) None)))))))))))))))))))))))) (let x27 (get x22) (let x28 (len x4) (let x29 (/ x27 x28) None)))))))))))))) None))))))))))))))))))))
```

4. GENERATED CODE
    1. The C++ code output by Lantern. This generated code is also available for inspection in the `gen` folder (in our case, it's the `module_runX.cpp` file).

```
==============================================================
========================GENERATED CODE========================
==============================================================
/*****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "lantern.h"
#include "module_runX.h"
using namespace std;
void x1(int x2) {
  float x3 = 0.0f;
  float* x14 = (float*)myMalloc(39200 * sizeof(float));
  for(int x16=0; x16 < 39200; x16++) {
    float x17 = (float)rand()/RAND_MAX;
    float x18 = x17 - 0.5f;
    float x19 = x18 * 1.0f;
    x14[x16] = x19;
  }
  float* x23 = (float*)myMalloc(50 * sizeof(float));
  for(int x25=0; x25 < 50; x25++) {
    float x26 = (float)rand()/RAND_MAX;
    float x27 = x26 - 0.5f;
    float x28 = x27 * 1.0f;
    x23[x25] = x28;
  }
  float* x32 = (float*)myMalloc(500 * sizeof(float));
  for(int x34=0; x34 < 500; x34++) {
    float x35 = (float)rand()/RAND_MAX;
    float x36 = x35 - 0.5f;
    float x37 = x36 * 1.0f;
    x32[x34] = x37;
  }
...
  return ;
}
int32_t entrypoint(int32_t  x0) {
}
/*******************************************/

```

5. EXECUTING CODE
    1. The output of the C++ code (generated by Lantern).

```
==============================================================
========================EXECUTING CODE========================
==============================================================

Train Epoch: 1 (6000/60000 (10%))   Loss: 2.282214
Train Epoch: 1 (12000/60000 (20%))  Loss: 1.521544
Train Epoch: 1 (18000/60000 (30%))  Loss: 1.237902
Train Epoch: 1 (24000/60000 (40%))  Loss: 1.034043
Train Epoch: 1 (30000/60000 (50%))  Loss: 0.916597
Train Epoch: 1 (36000/60000 (60%))  Loss: 0.822662
Train Epoch: 1 (42000/60000 (70%))  Loss: 0.753137
Train Epoch: 1 (48000/60000 (80%))  Loss: 0.698994
Train Epoch: 1 (54000/60000 (90%))  Loss: 0.657642
Train Epoch: 1 (60000/60000 (100%)) Loss: 0.614844
Train Epoch: 2 (6000/60000 (10%))   Loss: 0.259043
Train Epoch: 2 (12000/60000 (20%))  Loss: 0.251854
...
Train Epoch: 10 (54000/60000 (90%)) Loss: 0.106502
Train Epoch: 10 (60000/60000 (100%))    Loss: 0.103535
```

Finally, we have the timing results:

```
real    0m49.702s
user    0m48.432s
sys     0m0.404s
```

Despite the additional overhead associated with this metaprogramming and compilation, this runs nearly 3 times faster than the vanilla PyTorch code (clocking in well under a minute).

## Prerequisites

As always, the first step is to ensure all necessary prerequisites are installed. We detail the necessary tools and how to install them below.

### python3

Snek-LMS requires at least Python 3.5.

[Installing Python 3.5 for Linux](http://docs.python-guide.org/en/latest/starting/install3/linux/)

[Installing Python 3.5 for OSX](http://docs.python-guide.org/en/latest/starting/install3/osx/)

[Installing Python 3.5 for Windows](http://docs.python-guide.org/en/latest/starting/install3/win/)

### pip3

[Installing pip3](https://pip.pypa.io/en/stable/installing/)

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

### Learn More!

To learn more about Lantern, check out their website [here!](https://feiwang3311.github.io/Lantern/)

Interested in learning more about how Snek-LMS works? Read on!

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
