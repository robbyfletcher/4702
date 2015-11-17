
INCLUDE = -I/usr/X11R6/include/ -I/opt/local/include  \
  -I../include -I../../include

CUDAPATH = /usr/local/cuda

# -Xnvopencc -LIST:source=on

COMPILERFLAGS = -deviceemu -Xcompiler -Wall -G -Xcompiler -g  --ptxas-options=-v
COMPILERFLAGS = -Xcompiler -Wall --ptxas-options=-v \
   -Xcompiler -Wno-unused-function -use_fast_math
OPTFLAGS = -O3

CC = gcc
CXX = $(CUDAPATH)/bin/nvcc 

.SUFFIXES: .cu .cuh

CFLAGS = $(COMPILERFLAGS) $(INCLUDE)  -g -Xcompiler -Wno-strict-aliasing \
  -Xcompiler -Wno-parentheses 

MAGICKCXX_RAW := $(shell Magick++-config --cppflags --cxxflags)
# Remove openmp, which doesn't play well with nvcc.
MAGICKCXX := $(filter-out -fopenmp,$(MAGICKCXX_RAW))

CXXFLAGS = $(CFLAGS)  $(MAGICKCXX)

LIBRARIES =  -lX11 -lglut -lGL -lGLU -lm -lpthread  -lrt \
  $(shell Magick++-config --ldflags --libs)


default: BumperBalls

.cc.o:
	$(CXX) $(CXXFLAGS) -c $*.cc

.cu.o:
	$(CXX) $(CFLAGS) -c $*.cu

BumperBalls-kernel.o: Makefile BumperBalls.cuh BumperBalls-kernel.cu

BumperBalls: BumperBalls.cc shapes.h tiles.h Makefile BumperBalls.cuh BumperBalls-kernel.o
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $@.cc BumperBalls-kernel.o -o $@ $(LIBRARIES)
	$(CXX) $(CXXFLAGS) $@.cc -o $@-debug BumperBalls-kernel.o $(LIBRARIES)

