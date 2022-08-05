#include <iostream>
#include <cuda_runtime.h>
#include "CudaUtils.cuh"

using namespace std;

void chooseCudaDevice()
{
  // TODO: only call this once
  // Error or do nothing?
  cout << "Selecting the fastest GPU device...\n";
  int num_devices, device;
  checkCudaErrors(cudaGetDeviceCount(&num_devices));
  if (num_devices > 1)
  {
    int max_multiprocessors = 0, max_device = 0;
    for (device = 0; device < num_devices; device++)
    {
      cudaDeviceProp properties;
      checkCudaErrors(cudaGetDeviceProperties(&properties, device));
      if (max_multiprocessors < properties.multiProcessorCount)
      {
        max_multiprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, max_device));
    cout << "GPU device " << max_device << " selected; GPU device name: " << properties.name << endl;
    checkCudaErrors(cudaSetDevice(max_device));
  }
  else
  {
    cout << "Only one GPU device available, will use it (#0)\n";
    cudaDeviceProp properties;
    int device = 0;
    // checkCudaErrors(cudaGetDeviceProperties(&properties, device));
    // cout << "GPU device name: " << properties.name << endl;
    // cout << "Device Number: " << 0 << endl;
    // cout << "  Memory Clock Rate (KHz): " << properties.memoryClockRate << endl;
    // cout << "  Memory Bus Width (bits): " << properties.memoryBusWidth << endl;
    // cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*properties.memoryClockRate*(properties.memoryBusWidth/8)/1.0e6 << endl;
    // cout << "  Compute capability: " << properties.major << "." << properties.minor << endl;
    checkCudaErrors(cudaSetDevice(device));
  }
}
