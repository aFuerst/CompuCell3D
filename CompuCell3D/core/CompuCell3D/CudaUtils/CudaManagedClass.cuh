#include <cuda.h>

class CudaManaged {
/*
    Base class to freely handle data allocation for objects used in CUDA unified memory
    Classes should only have pointer or data types as members

    TODO: implement move and copy constructors?
*/

public:
  private:
    // any usages of these should cause compile error
    CudaManaged() { }
    ~CudaManaged() {}
    // Copy constructor.
    CudaManaged(const CudaManaged &other) {}
    // Copy assignment operator.
    CudaManaged &operator=(const CudaManaged &other) {}
    // move operator
    CudaManaged &operator=(CudaManaged &&other) {}

  public:
    void *operator new(size_t len)
    {
      void *ptr;
      checkCudaErrors(cudaMallocManaged(&ptr, len));
      checkCudaErrors(cudaDeviceSynchronize());
      return ptr;
    }

    void operator delete(void *ptr)
    {
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaFree(ptr));
    }
};
