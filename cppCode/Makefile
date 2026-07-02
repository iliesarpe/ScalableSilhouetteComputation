CC = g++
CFLAGS = -v --std=c++2a -g -fopenmp -O3 -march=native#-fsanitize=address 
#CFLAGS = -I.
#DEPS = structures.h
FOLLIBS = ${PWD}/libs/
EXEC = efficientSilh
#OBJ = structures.o clustcoeff.o 

#%.o: %.cpp $(DEPS)
#	$(CC) -o $@ $< $(CFLAGS)


all: main.cpp ${FOLLIBS}structures.h structures.o ${FOLLIBS}algorithms.h algorithms.o
	$(CC) ${CFLAGS} main.cpp structures.o algorithms.o -o ${EXEC} -Ilibs/ -I/usr/include/hdf5/serial -I./import/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5

structures.o: ${FOLLIBS}structures.cpp
	$(CC) -c ${CFLAGS} ${FOLLIBS}structures.cpp -Ilibs/ -I/usr/include/hdf5/serial -I./import/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5

algorithms.o: ${FOLLIBS}structures.h structures.o ${FOLLIBS}algorithms.cpp
	$(CC) -c ${CFLAGS} ${FOLLIBS}algorithms.cpp -Ilibs/ -I/usr/include/hdf5/serial -I./import/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5

clean:
	rm ${EXEC} *.o
	rm -rf *.dSYM
#g++ \
#--std=c++2a \
#-g \
#-O3 \
#-o $@ \
#$< \
