NAME=snek
CXX=g++
INCLUDE=-I/usr/include/python2.7 -I/usr/lib/python2.7

all:
	swig -c++ -python ${NAME}.i
	g++ -std=c++0x -c -fPIC ${NAME}.cpp
	g++ -c -fPIC ${NAME}_wrap.cxx ${INCLUDE}
	g++ -shared -Wl,-soname,_${NAME}.so -o _${NAME}.so ${NAME}.o ${NAME}_wrap.o

clean:
	rm -f *.cxx
	rm -f *.o
	rm -f *.py
	rm -f *.so
