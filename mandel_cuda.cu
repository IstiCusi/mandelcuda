#include <cuda_runtime.h>
__device__ float mapIterationsToWavelength(int iterations, int maxIter, float minWavelength, float maxWavelength) {
    float logIter = logf(iterations);
    float logMaxIter = logf(maxIter);
    float normalizedLogIter = (logIter - logf(1)) / (logMaxIter - logf(1));
    float expScale = exp(normalizedLogIter * log(maxWavelength / minWavelength));
    float wavelength = minWavelength * expScale;
    return wavelength;
}

__global__ void mandelbrotKernel(int *output, float *wavelengths, float xmin, float xmax, float ymin, float ymax, int width, int height, int maxIter, float minWavelength, float maxWavelength) {
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
        wavelengths[idy * width + idx] = mapIterationsToWavelength(iteration, maxIter, minWavelength, maxWavelength);
    }
}

extern "C" void computeMandelbrot(int *output         , float *wavelengths     ,
                                  float xmin          , float xmax             ,
                                  float ymin          , float ymax             ,
                                  int width           , int height             ,
                                  int maxIter         ,
                                  float minWavelength , float maxWavelength) {
    int *d_output;
    float *d_wavelengths;
    size_t size = width * height * sizeof(int);
    size_t sizeWavelengths = width * height * sizeof(float);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_wavelengths, sizeWavelengths);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    mandelbrotKernel<<<dimGrid, dimBlock>>>(d_output, d_wavelengths, xmin, xmax, ymin, ymax, width, height, maxIter, minWavelength, maxWavelength);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost); // Blocks automatically
    cudaMemcpy(wavelengths, d_wavelengths, sizeWavelengths, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_wavelengths);
}

