### LSU EE 4702-1 (Fall 2013), GPU Programming
##
##  Makefile for Classroom Demos
##
## $Id:$


COURSE_INCLUDE_DIR = ../include
CUDAPATH = /usr/local/cuda



OPT_FLAGS =  -O3

CXX = g++

CUDUMP = $(CUDAPATH)/bin/cuobjdump
CUXX = $(CUDAPATH)/bin/nvcc

.SUFFIXES: .cu .cuh

INCLUDE = -I$(COURSE_INCLUDE_DIR)

PACKAGE_CXXFLAGS = $(shell Magick++-config --cppflags --cxxflags)

CXXFLAGS = $(INCLUDE) $(PACKAGE_CXXFLAGS) $(OPTFLAGS) \
	 -I$(CUDAPATH)/include \
	 -g -Wall -Wno-strict-aliasing

LINKFLAGS = $(OPTFLAGS)

LIBRARIES =  -lX11 -lXi -lglut -lGL -lGLU -lm -lpthread  -lrt \
  $(shell Magick++-config --ldflags)
LIBDIR  =

default: all
#default: demo-11-a-comp-shader
all:playground

INCLUDES_BASE = util.h glextfuncs.h coord.h shader.h pstring.h misc.h \
	        gl-buffer.h texture-util.h cuda-util.h

INCLUDES = $(patsubst %,$(COURSE_INCLUDE_DIR)/%,$(INCLUDES_BASE)) \
	 shapes.h Makefile


CUFLAGS = -g -O3 -Xcompiler -Wall -Xcompiler -Wno-unused-function \
 -Xcompiler -Wno-parentheses \
 --ptxas-options=-v  -use_fast_math --gpu-architecture=sm_20 \
 $(INCLUDE)

%.o: %.cc Makefile $(INCLUDES)
	$(CXX) $(CXXFLAGS) -c $*.cc

%.o: %.cu Makefile
	$(CUXX) $(CUFLAGS) -c $*.cu
	$(CUXX) $(CUFLAGS) -ptx $*.cu -o $*.ptx
	$(CUXX) $(CUFLAGS) -cubin $*.cu -o $*.cubin
	$(CUDUMP) -sass $*.cubin > $*.sass

playground.o:  $(INCLUDES) playground-graphics.cc
playground: playground.o
	$(CXX) $(LINKFLAGS) -o $@ $(LIBDIR) $^ $(LIBRARIES)

clean:
	/bin/rm -f  *.o *~ *.sass *.cubin *.ptx
