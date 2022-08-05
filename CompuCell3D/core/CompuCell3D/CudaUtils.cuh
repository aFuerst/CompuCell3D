#ifndef CUDAUTILS_CUH
#define CUDAUTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    char buff[256];
    sprintf(buff, "%s(%i) : CUDA Runtime API error %d: %s", file, line, (int)err, cudaGetErrorString(err));
    fprintf(stderr, "%s\n", buff);
    // ASSERT_OR_THROW(buff, err==cudaSuccess);

    exit(-1);
  }
}

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void chooseCudaDevice();

#endif // CUDAUTILS_CUH