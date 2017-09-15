all: test.cu util.h
	 nvcc -std=c++11 -arch=compute_35 -code sm_35 test.cu -o transpose