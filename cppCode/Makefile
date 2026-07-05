CC      = g++
CFLAGS  = --std=c++2a -g -O3 -march=native
FOLLIBS = ${PWD}/libs/
EXEC    = efficientSilh

# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  # macOS — install dependencies with Homebrew:
  #   brew install hdf5 libomp nlohmann-json
  BREW        := $(shell brew --prefix 2>/dev/null || echo /usr/local)
  HDF5_PREFIX := $(shell brew --prefix hdf5   2>/dev/null || echo $(BREW))
  LIBOMP      := $(shell brew --prefix libomp 2>/dev/null || echo $(BREW))

  HDF5_INC  = -I$(HDF5_PREFIX)/include
  HDF5_LIB  = -L$(HDF5_PREFIX)/lib
  # Apple Clang does not bundle OpenMP; link against Homebrew libomp.
  # Override with CC=g++-14 if using Homebrew GCC (uses -fopenmp directly).
  OMP_FLAGS = -Xclang -fopenmp
  OMP_INC   = -I$(LIBOMP)/include
  OMP_LIB   = -L$(LIBOMP)/lib -lomp
  # Covers nlohmann-json installed via: brew install nlohmann-json
  EXTRA_INC = -I$(BREW)/include
else
  # Linux (Ubuntu/Debian) — install dependencies with apt:
  #   apt install g++ libhdf5-dev
  # nlohmann/json: placed at /usr/local/include/nlohmann/json.hpp
  #   (see image.def or: apt install nlohmann-json3-dev)
  HDF5_INC  = -I/usr/include/hdf5/serial
  HDF5_LIB  = -L/usr/lib/x86_64-linux-gnu/hdf5/serial
  OMP_FLAGS = -fopenmp
  OMP_INC   =
  OMP_LIB   =
  EXTRA_INC =
endif

CFLAGS  += $(OMP_FLAGS)
ALL_INC  = -Ilibs/ -I./import/ $(HDF5_INC) $(OMP_INC) $(EXTRA_INC)
ALL_LIBS = $(HDF5_LIB) $(OMP_LIB) -lhdf5

# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
all: main.cpp $(FOLLIBS)structures.h structures.o $(FOLLIBS)algorithms.h algorithms.o
	$(CC) $(CFLAGS) main.cpp structures.o algorithms.o -o $(EXEC) $(ALL_INC) $(ALL_LIBS)

structures.o: $(FOLLIBS)structures.cpp
	$(CC) -c $(CFLAGS) $(FOLLIBS)structures.cpp $(ALL_INC)

algorithms.o: $(FOLLIBS)algorithms.h structures.o $(FOLLIBS)algorithms.cpp
	$(CC) -c $(CFLAGS) $(FOLLIBS)algorithms.cpp $(ALL_INC)

clean:
	rm -f $(EXEC) *.o
	rm -rf *.dSYM
