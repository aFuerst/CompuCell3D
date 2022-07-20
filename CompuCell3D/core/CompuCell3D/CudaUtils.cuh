#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <stdio.h>
#include <cuda.h>

inline void __checkCudaErrors(cudaError_enum err, const char *file, const int line);

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void chooseCudaDevice();

#endif // CUDAUTILS_H