#include <iostream>
#include <CompuCell3D/Simulator.h>
#include "FieldExtractor_CUDA.cuh"
#include "FieldExtractor.h"
// #include <CompuCell3D/steppables/PDESolvers/CUDA/CUDAUtilsHeader.h>
#include <CompuCell3D/CudaUtils/CudaUtils.h>
#include <cuda.h>
#include "FieldStorage.h"

using namespace std;
using namespace CompuCell3D;

FieldExtractor_CUDA::FieldExtractor_CUDA()
{
  fsPtr = nullptr;
  potts = nullptr;
  sim = nullptr;
  cell_field = nullptr;
  con_field = nullptr;
  cout << "constructing FieldExtractor_CUDA" << endl;
}

void FieldExtractor_CUDA::init(Simulator *_sim)
{
  cout << "initializing FieldExtractor_CUDA" << endl;
  sim = _sim;
  potts = sim->getPotts();
  FieldExtractor::init(_sim);

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

  alloc();
}

void FieldExtractor_CUDA::alloc() {
  cout << "alloc FieldExtractor_CUDA" << endl;

  if (! fsPtr) {
    cerr << "FieldStorage pointer is null in FieldExtractor_CUDA::alloc!!!" << endl;
    return;
  }
  auto dim = fsPtr->getDim();
  size_t field_size = dim.x * dim.y * dim.z * sizeof(float);

  float *cell_f=nullptr;
  if (cell_field)
  {
    cout << "FieldExtractor_CUDA cell_field is already assigned!!!" << endl;
    return;
  }
  cout << "FieldExtractor_CUDA cudaMalloc cell_field!!! " << field_size << endl;
  // checkCudaErrors(cudaPeekAtLastError());
  // checkCudaErrors(cudaMalloc((void **)&cell_f, field_size));

  float *con_f=nullptr;
  if (con_field)
  {
    cout << "FieldExtractor_CUDA con_field is already assigned!!!" << endl;
    return;
  }
  cout << "Peek error:" << endl;
  checkCudaErrors(cudaPeekAtLastError());

  cout << "FieldExtractor_CUDA cudaMalloc con_field!!! " << field_size << endl;
  checkCudaErrors(cudaMalloc(&con_f, field_size));

  cout << "done alloc FieldExtractor_CUDA" << endl;
  cell_field = cell_f;
  // con_field = con_f;

  float* dev_prt = moveConFieldToDevice("");
  if (dev_prt == NULL) {
    cout << "con feld storage succeeded" << endl;
  }
  else {
    cout << "con feld storage failed" << endl;
  }
}

void FieldExtractor_CUDA::setFieldStorage(FieldStorage *_fsPtr)
{
  cout << "FieldExtractor_CUDA setting field storage" << endl;
  FieldExtractor::setFieldStorage(_fsPtr);
  this->fsPtr = _fsPtr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractor_CUDA::~FieldExtractor_CUDA(){
  cout << "deconstruct FieldExtractor_CUDA" << endl;
  if (cell_field)
  {
    cout << "freeing cell_field" << endl;
    checkCudaErrors(cudaFree(cell_field));
  }
  if (con_field)
  {
    cout << "freeing con_field" << endl;
    checkCudaErrors(cudaFree(con_field));
  }
}

float* FieldExtractor_CUDA::moveCellFieldToDevice(std::string _conFieldName)
{
}

float* FieldExtractor_CUDA::moveConFieldToDevice(std::string _conFieldName) {
  float* dev_prt=NULL;
  auto dim = fsPtr->getDim();
  size_t field_size = dim.x * dim.y * dim.z * sizeof(float);
  Field3D<float> *conFieldPtr=NULL;

  std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
  std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}
    
	if(!conFieldPtr)
		return NULL;

  checkCudaErrors(cudaMemcpy(con_field, conFieldPtr, field_size, cudaMemcpyHostToDevice));
  return dev_prt;
}

void FieldExtractor_CUDA::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillCellFieldData2D" << endl;
  // FieldExtractor::fillCellFieldData2D(_cellTypeArrayAddr, _plane, _pos);
}

void FieldExtractor_CUDA::fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillCellFieldData2DCartesian" << endl;
  // FieldExtractor::fillCellFieldData2DCartesian(_cellTypeArrayAddr, _cellsArrayAddr, _pointsArrayAddr, _plane, _pos);
}

void FieldExtractor_CUDA::fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillBorderData2D" << endl;
  // FieldExtractor::fillBorderData2D(_pointArrayAddr, _linesArrayAddr, _plane, _pos);
}

void FieldExtractor_CUDA::fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillClusterBorderData2D" << endl;
}

bool FieldExtractor_CUDA::fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillConFieldData2D" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillConFieldData2DCartesian" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillScalarFieldData2D" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillScalarFieldData2DCartesian" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillScalarFieldCellLevelData2D" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillScalarFieldCellLevelData2DCartesian" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillVectorFieldData2D" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillVectorFieldCellLevelData2D" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName)
{
  cout << "FieldExtractor_CUDA fillVectorFieldData3D" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillVectorFieldData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName) {
  cout << "FieldExtractor_CUDA fillVectorFieldData3DHex" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillVectorFieldCellLevelData2DHex" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName)
{
  cout << "FieldExtractor_CUDA fillVectorFieldCellLevelData3D" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillVectorFieldCellLevelData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName)
{
  cout << "FieldExtractor_CUDA fillVectorFieldCellLevelData3DHex" << endl;
  return false;
}

bool FieldExtractor_CUDA::fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec) {
  cout << "FieldExtractor_CUDA fillScalarFieldData3D" << endl;
  return false;
}

std::vector<int> FieldExtractor_CUDA::fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr, bool extractOuterShellOnly)
{
  cout << "FieldExtractor_CUDA fillCellFieldData3D" << endl;
  // auto ret = FieldExtractor::fillCellFieldData3D(_cellTypeArrayAddr, _cellIdArrayAddr, extractOuterShellOnly);
  // cout << "DONE FieldExtractor_CUDA fillCellFieldData3D" << endl;
  auto ret = std::vector<int>();
  return ret;
}
bool FieldExtractor_CUDA::fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec)
{
  cout << "FieldExtractor_CUDA fillConFieldData3D" << endl;
  return false;
}

void FieldExtractor_CUDA::fillCellFieldData2DHex_old(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillCellFieldData2DHex_old" << endl;
}
void FieldExtractor_CUDA::fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillCellFieldData2DHex" << endl;
}
void FieldExtractor_CUDA::fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillBorderData2DHex" << endl;
}
void FieldExtractor_CUDA::fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillClusterBorderData2DHex" << endl;
}

void FieldExtractor_CUDA::fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillCentroidData2D" << endl;
}
bool FieldExtractor_CUDA::fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillConFieldData2DHex" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillScalarFieldData2DHex" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillScalarFieldCellLevelData2DHex" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec) {
  cout << "FieldExtractor_CUDA fillScalarFieldCellLevelData3D" << endl;
  return false;
}
bool FieldExtractor_CUDA::fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillVectorFieldData2DHex" << endl;
  return false;
}
