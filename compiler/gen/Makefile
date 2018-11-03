CXX=g++
INCLUDE=`python3-config --includes`

all:
	swig -c++ -python ${MODULE_NAME}.i
	g++ -O1 -std=c++11 -c -fPIC ${MODULE_NAME}.cpp
	g++ -O3 -c -fPIC ${MODULE_NAME}_wrap.cxx ${INCLUDE}
	g++ -O3 -shared -Wl,-soname,_${MODULE_NAME}.so -o _${MODULE_NAME}.so ${MODULE_NAME}.o ${MODULE_NAME}_wrap.o

clean:
	rm *.cxx
	rm *.o