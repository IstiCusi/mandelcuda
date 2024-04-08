CC=gcc
NVCC=nvcc
CFLAGS=-Iinclude -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lraylib -lGL -lm -lpthread -ldl -lrt -lX11


TARGET=mandelbrot

# Quelldateien
CUDA_SRC=mandel_cuda.cu
C_SRC=mandelbrot.c

# Objektdateien
CUDA_OBJ=$(CUDA_SRC:.cu=.o)
C_OBJ=$(C_SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(CUDA_OBJ) $(C_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(CFLAGS)

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(TARGET) $(CUDA_OBJ) $(C_OBJ)

.PHONY: all clean

