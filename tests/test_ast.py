from pylms import ast

@ast
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

def test_power():
    assert(power.original(2,3) == 8)
    assert(power(2,3) == 8)

def test_power_code():
    assert(power.code == """(def power (b x) (begin (if (== x 0) (return 1) (return (* b (power b (- x 1)))))))""")

def test_power_Ccode():
    sol = """/*****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "power.h"
using namespace std;
int x1(int x2, int x3) {
  bool x4 = x3 == 0;
  int32_t x10;
  if (x4) {
    x10 = 1;
  } else {
    int32_t x5 = x3 - 1;
    int32_t x7 = x1(x2,x5);
    int32_t x8 = x2 * x7;
    x10 = x8;
  }
  return x10;
}

int32_t entrypoint(int32_t  x0) {
  return 0;
}
/*******************************************/
"""

    assert(power.Ccode == sol)

def test_power_bin(): # TODO create name
    import power as powermod
    assert(powermod.x1(2,3) == 8)



@ast
def ifelse(x):
    numpy.zeros(5)
    if x == 0:
        print("Hello")
    else:
        print("world!")
    return x

def test_ifelse_code():
    assert(ifelse.code == """(def ifelse (x) (begin (call numpy zeros 5) (if (== x 0) (print "Hello") (print "world!")) (return x)))""")

def test_ifelse_Ccode():
    sol = """/*****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ifelse.h"
using namespace std;
int x1(int x2) {
  int32_t* x3 = (int32_t*)malloc(5 * sizeof(int32_t));
  bool x4 = x2 == 0;
  int32_t x9;
  if (x4) {
    printf("%s\\n","Hello");
    x9 = 1;
  } else {
    printf("%s\\n","world!");
    x9 = 1;
  }
  return x2;
}

int32_t entrypoint(int32_t  x0) {
  return 0;
}
/*******************************************/
"""

    assert(ifelse.Ccode == sol)

def test_ifelse_bin(): # TODO
    import ifelse as ifelsemod
    assert(ifelsemod.x1(8) == 8)
