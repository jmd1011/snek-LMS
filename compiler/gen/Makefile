CXX=g++
INCLUDE=`python3-config --includes`

all:
	swig3.0 -c++ -python ${MODULE_NAME}.i
	g++ -std=c++0x -c -fPIC ${MODULE_NAME}.cpp
	g++ -c -fPIC ${MODULE_NAME}_wrap.cxx ${INCLUDE}
	g++ -shared -Wl,-soname,_${MODULE_NAME}.so -o _${MODULE_NAME}.so ${MODULE_NAME}.o ${MODULE_NAME}_wrap.o

clean:
	rm *.cxx
	rm *.o
