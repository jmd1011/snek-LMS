CXX=g++
INCLUDE=`python3-config --includes`

all:
	swig -c++ -python ${MODULE_NAME}.i
	g++ -O3 -Wno-pointer-arith -std=c++0x -c -fPIC -I. ${MODULE_NAME}.cpp
	g++ -O3 -c -fPIC ${MODULE_NAME}_wrap.cxx ${INCLUDE}
	g++ -O3 -shared -o _${MODULE_NAME}.so ${MODULE_NAME}.o ${MODULE_NAME}_wrap.o

clean:
	rm -f *.c*
	rm -f *.o
	rm -f module_*.h
	rm -f *.py*
	rm -f *.so
	rm -f *.i
	rm -rf __pycache__
