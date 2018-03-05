NAME=snek
CXX=g++
INCLUDE=`python3-config --includes`

all:
	swig3.0 -c++ -python ${NAME}.i
	g++ -std=c++0x -c -fPIC ${NAME}.cpp
	g++ -c -fPIC ${NAME}_wrap.cxx ${INCLUDE}
	g++ -shared -Wl,-soname,_${NAME}.so -o _${NAME}.so ${NAME}.o ${NAME}_wrap.o

clean:
	rm -f *.cxx
	rm -f *.o
	rm -f *.py
	rm -f *.so
