from pylms import ast


@ast
def power(b, x):
    if (x == 0): return 1
    else: return b * power(b, x-1)

def test_power():
    assert(power.original(2,3) == 8)
    assert(power(2,3) == 8)

def test_power_code():
    assert(power.code == """(def power (b x) ((if (== x 0) (return 1) (return (* b (power b (- x 1))))))) (power 4 2)""")

def test_power_Ccode():
    sol = """/*****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
using namespace std;
int32_t entrypoint(int32_t  x0) {
  function<int32_t(int32_t,int32_t)> x1 = [&](int32_t x2,int32_t x3) {
    int32_t x4 = x2;
    int32_t x5 = x3;
    int32_t x7 = x5 - 1;
    int32_t x9 = x1(x4,x7);
    bool x6 = x5 == 0;
    int32_t x11;
    if (x6) {
      x11 = 1;
    } else {
      int32_t x10 = x4 * x9;
      x11 = x10;
    }
    return x11;
  };
  int32_t x14 = x1(4,2);
  return x14;
}
/*******************************************/"""

    assert(power.Ccode == sol)

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

