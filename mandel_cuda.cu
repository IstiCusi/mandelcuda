#include <cuda_runtime.h>

__global__ void mandelbrotKernel(int *output, float xmin, float xmax, float ymin, float ymax, int width, int height, int maxIter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        float x0 = xmin + (float)idx / width * (xmax - xmin);
        float y0 = ymin + (float)idy / height * (ymax - ymin);
        float x = 0.0f, y = 0.0f;
        int iteration = 0;
        while (x*x + y*y <= 4 && iteration < maxIter) {
            float xtemp = x*x - y*y + x0;
            y = 2*x*y + y0;
            x = xtemp;
            iteration++;
        }
        output[idy * width + idx] = iteration;
    }
}

extern "C" void computeMandelbrot(int *output, float xmin, float xmax, float ymin, float ymax, int width, int height, int maxIter) {
    int *d_output;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_output, size);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    mandelbrotKernel<<<dimGrid, dimBlock>>>(d_output, xmin, xmax, ymin, ymax, width, height, maxIter);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}

