# MandelCuda

## Description

MandelCuda is a very minimalistic show case of CUDA for the calculation of the Mandelbrot set
and its boundaries.

![First Calculation without spectral mapping](docs/assets/5.jpeg)
![Log-Log Distribution](docs/assets/d1.jpeg)

## Mathematical Overview

The iteration inside the Mandelbrot set generation loop, as found in the [MandelbrotKernel](https://github.com/IstiCusi/mandelcuda/blob/main/mandel_cuda.cu#L21-L26) 
is mathematically very simple and represented by:

The iteration inside the Mandelbrot set generation loop is mathematically represented by:

$$
z_{n+1} = z_n^2 + c
$$

with \( z_0 = 0 \) and \( c = x_0 + iy_0 \), where \( x_0 \) and \( y_0 \) are the real and imaginary parts of \( c \), calculated from pixel coordinates. During each iteration, we perform:

$$
\text{Real}(z_{n+1}) = x^2 - y^2 + x_0
$$
$$
\text{Imaginary}(z_{n+1}) = 2xy + y_0
$$

The process continues until either the magnitude of \( z_n \) squared (\( x^2 + y^2 \)) exceeds 4, indicating the point does not belong to the Mandelbrot set, or the maximum number of iterations is reached.


This iteration continues until either the magnitude of \( z_n \) squared (\( x^2 + y^2 \)) exceeds 4,
indicating the point does not belong to the Mandelbrot set, or the maximum number of iterations is reached.


## Installation

Compile the mandelcute-iness using `make`

## Contributing

Any contribution is welcome!

## License

This project is open-source and is licensed under the MIT License. Please see the 
[LICENSE](LICENSE.md) file for detailed licensing information.


## TODOS

- Complete API documention 
- Extend about better coloring schemes and zooming etc
- The coloring is based on a heuristic assumption of the distribution of iterations 
  (A better color mapping could be done in sequence)
