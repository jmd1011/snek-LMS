from pylms import ast


@ast
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

def test_power():
    assert(power.original(2,3) == 8)
    assert(power(2,3) == 8)

def test_power_code():
    assert(power.code == """(def power (b x) ((if (== x 0) (return 1) (return (* b (power b (- x 1)))))))""")

def test_power_Ccode():
    sol = """/*****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
using namespace std;
int32_t x1(int32_t x2, int32_t x3);

int32_t x1(int32_t x2, int32_t x3) {
  int32_t x5 = x3 - 1;
  int32_t x7 = x1(x2,x5);
  bool x4 = x3 == 0;
  int32_t x9;
  if (x4) {
    x9 = 1;
  } else {
    int32_t x8 = x2 * x7;
    x9 = x8;
  }
  return x9;
}

int32_t entrypoint(int32_t  x0) {
  return 0;
}
/*******************************************/
"""

    #assert(power.Ccode == sol)

@ast
def ifelse(x):
    numpy.zeros(5)
    if x == 0:
        print("Hello")
    else:
        print("world!")
    return x

def test_ifelse_code():
    assert(ifelse.code == """(def ifelse (x) ((call numpy zeros 5) (if (== x 0) (print "Hello") (print "world!")) (return x)))""")

