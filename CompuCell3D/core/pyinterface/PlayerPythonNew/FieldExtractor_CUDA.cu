#include <iostream>
#include <CompuCell3D/Simulator.h>
#include "FieldExtractor_CUDA.cuh"
#include "FieldExtractor.h"
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
// #include <CompuCell3D/steppables/PDESolvers/CUDA/CUDAUtilsHeader.h>
#include <CompuCell3D/CudaUtils.cuh>
#include <cuda_runtime.h>
#include "FieldStorage.h"

using namespace std;
using namespace CompuCell3D;

FieldExtractor_CUDA::FieldExtractor_CUDA()
{
  fsPtr = nullptr;
  potts = nullptr;
  sim = nullptr;
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
}

void FieldExtractor_CUDA::pointOrder(std::string _plane, int* order){
	for (int i = 0; i <_plane.size(); ++i){
		_plane[i]=tolower(_plane[i]);
	}
	order[0] =0;
	order[1] =1;
	order[2] =2;
	if (_plane == "xy") {
		order[0] =0;
		order[1] =1;
		order[2] =2;            
	}
	else if (_plane == "xz") {
		order[0] =0;
		order[1] =2;
		order[2] =1;            
  }
	else if( _plane == "yz") { 
		order[0] =2;
		order[1] =0;
		order[2] =1;            
	}
}
void FieldExtractor_CUDA::dimOrder(std::string _plane, int* order){
	for (int i = 0  ; i <_plane.size() ; ++i){
		_plane[i]=tolower(_plane[i]);
	}
	order[0] =0;
	order[1] =1;
	order[2] =2;
	if (_plane == "xy"){
		order[0] =0;
		order[1] =1;
		order[2] =2;            
	}
	else if (_plane == "xz"){
		order[0] =0;
		order[1] =2;
		order[2] =1;            
  }
	else if( _plane == "yz"){ 
		order[0] =1;
		order[1] =2;
		order[2] =0;            
  }
}

void FieldExtractor_CUDA::fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillCellFieldData2D" << endl;
  // FieldExtractor::fillCellFieldData2D(_cellTypeArrayAddr, _plane, _pos);
}

__device__
unsigned int index_dev(int _x, int _y, int _z, int x_dim, int y_dim) {
  //start indexing from 0'th element but calculate index based on increased lattice dimmension
  return _x + (_y + _z * y_dim) * x_dim;
}


__global__
void fillCellFieldData2DCartesian_gpu(int* cellsArray_out, int* cell_types_out, double* cellCoords_out, FieldExtractParams_t* params) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int bz = 0; //simulated blockIdx.z
  int DIMX = params->dim[0];
  int DIMY = params->dim[1];
  int DIMZ = params->dim[2];

  int bz_max = DIMZ / 16;
  int ptArr[3];

  for (bz = 0; bz < bz_max; ++bz) {
    int x = bx * 16 + tx;
    int y = by * 16 + ty;
    int z = bz * 16 + tz;

    unsigned int idx = index_dev(x,y,z, params->dim[0], params->dim[1]);

    CellG* cell = params->cellField[idx];
    cell_types_out[idx] = (int) cell->type;

    // Coordinates3D<double> coords(ptVec[0], ptVec[1], 0);
    // for (int idx = 0; idx < 4; ++idx)
    // {
    //   Coordinates3D<double> cartesianVertex = cartesianVertices[idx];
    //   data[idx*2] = cartesianVertex.x + ;
    //   data[(idx*2) + 1] = cartesianVertex.y;
    // }

    int arrPos = idx * 5;
    int cellPos = idx * 4;
    cellsArray_out[arrPos + 0] = 4;
    cellsArray_out[arrPos + 1] = cellPos + 0;
    cellsArray_out[arrPos + 2] = cellPos + 1;
    cellsArray_out[arrPos + 3] = cellPos + 2;
    cellsArray_out[arrPos + 4] = cellPos + 3;
  }
}

void FieldExtractor_CUDA::fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos) {
  cout << "FieldExtractor_CUDA fillCellFieldData2DCartesian" << endl;
  vtkIntArray *_cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
  vtkPoints *_pointsArray = (vtkPoints *)_pointsArrayAddr;
  vtkCellArray * _cellsArray = (vtkCellArray*)_cellsArrayAddr;

  Field3D<CellG*> * cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

  FieldExtractParams_t params;
  params.cellField = cellFieldG->getPtr();
  vector<int> fieldDimVec(3, 0);
  fieldDimVec[0] = fieldDim.x;
  fieldDimVec[1] = fieldDim.y;
  fieldDimVec[2] = fieldDim.z;

  pointOrder(_plane, params.pointOrderVec);
  dimOrder(_plane, params.dimOrderVec);
  params.dim[0] = fieldDimVec[params.dimOrderVec[0]];
  params.dim[1] = fieldDimVec[params.dimOrderVec[1]];
  params.dim[2] = fieldDimVec[params.dimOrderVec[2]];
  params.pos = _pos;

  int* d_cellsArray;
  int* d_cellsType;
  double* d_cellCoords;
  int numPoints = params.dim[0] * params.dim[1] * params.dim[2];
  size_t cellsArraySize = numPoints * sizeof(unsigned char);
  checkCudaErrors(cudaMalloc((void**)&d_cellsArray, cellsArraySize*5));
  checkCudaErrors(cudaMalloc((void**)&d_cellsType, cellsArraySize));
  checkCudaErrors(cudaMalloc((void**)&d_cellCoords, numPoints*2*4*sizeof(double)));
  FieldExtractParams_t* params_device;
  // checkCudaErrors(cudaHostGetDevicePointer((void **) &params_device, (void *)&params, 0));
  checkCudaErrors(cudaMalloc((void**)&params_device, sizeof(FieldExtractParams_t)));
  checkCudaErrors(cudaMemcpy((void *)params_device, (void *)&params, sizeof(FieldExtractParams_t), cudaMemcpyHostToDevice));

  dim3 threads(16, 16, 16);
  dim3 grid(params.dim[0] / threads.x, params.dim[1] / threads.y);

  fillCellFieldData2DCartesian_gpu<<< grid, threads >>>(d_cellsArray, d_cellsType, d_cellCoords, params_device);
  vtkIdType *_cellsArrayWritePtr;
  int* h_cellsType;
  #pragma omp parallel
  {
    #pragma omp sections
    {
      #pragma omp section
      {

        _cellsArrayWritePtr = _cellsArray->WritePointer(numPoints, numPoints*5);
        checkCudaErrors(cudaMemcpy(d_cellsArray, _cellsArrayWritePtr, cellsArraySize, cudaMemcpyDeviceToHost));
      }
      #pragma omp section
      {
        _cellTypeArray->SetNumberOfValues(numPoints);
        h_cellsType = new int[cellsArraySize*5];
        checkCudaErrors(cudaMemcpy(d_cellsType, h_cellsType, cellsArraySize*5, cudaMemcpyDeviceToHost));
      }
      #pragma omp section
      {
        _pointsArray->SetNumberOfPoints(numPoints*4);
      }
    }
    #pragma omp once
    cudaDeviceSynchronize();
  
    #pragma omp for schedule(static)
    for (int j = 0; j < numPoints; ++j) {
      int cellPos = j*4;
			for (int idx = 0; idx < 4; ++idx) {
			  // Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords; 
 			  // _pointsArray->SetPoint(cellPos + idx, cartesianVertex.x, cartesianVertex.y, 0.0);
			}
      _cellTypeArray->SetValue(j, h_cellsType[j]);
    }
  }
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
