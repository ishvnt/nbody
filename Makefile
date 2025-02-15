CC = g++
NVCC = nvcc
INC = -Iinc
DBG = -g
ARCH = -arch=native

all: nbody

nbody: build/display.o build/kernels.o src/main.cu
	$(NVCC) $(DBG) $(INC) -o nbody src/main.cu build/display.o build/kernels.o $(ARCH) -lSDL2

build/display.o: src/display.cu
	mkdir -p build
	$(NVCC) $(DBG) $(INC) -c src/display.cu -o build/display.o $(ARCH)

build/kernels.o: src/kernels.cu
	$(NVCC) $(DBG) $(INC) -c src/kernels.cu -o build/kernels.o $(ARCH)
	
run: nbody
	./nbody

clean:
	rm nbody
	rm -r build