CC = g++
NVCC = nvcc
INC = -Iinc
DBG = -g
ARCH = -arch=native

all: main

main: display.o kernels.o src/main.cu inc/main.h
	$(NVCC) $(DBG) $(INC) -o main src/main.cu display.o kernels.o $(ARCH) -lSDL2

display.o: src/display.cu
	$(NVCC) $(DBG) $(INC) -c src/display.cu -o display.o $(ARCH)

kernels.o: src/kernels.cu
	$(NVCC) $(DBG) $(INC) -c src/kernels.cu -o kernels.o $(ARCH)
	
run: main
	./main

clean:
	rm *.o main