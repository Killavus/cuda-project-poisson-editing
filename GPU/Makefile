PROGNAME=cuda_paste
CC=g++
CPPFLAGS=`pkg-config --cflags MagickWand` -I.

NVCC=nvcc
NVFLAGS=-g -G -O0 -I. -I$(CUDA_DIR)/include -I$(CUDA_DIR)/samples/common/inc -arch=sm_61
LDFLAGS=`pkg-config --libs MagickWand`
OBJS=main.o src/host_image.o src/device_util.cu.o src/naive.cu.o src/laplacian.cu.o src/poisson.cu.o
.SUFFIXES: .cpp .c .h .y. .l .o .cu
.PHONY: clean main

%.cu.o: %.cu
	$(NVCC) $(NVFLAGS) -dc $< -o $@

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@ 

main: $(OBJS) 
	$(NVCC) $(OBJS) $(NVFLAGS) $(LDFLAGS) -o $(PROGNAME) 

clean:
	rm -f *.o **/*.o $(PROGNAME)
