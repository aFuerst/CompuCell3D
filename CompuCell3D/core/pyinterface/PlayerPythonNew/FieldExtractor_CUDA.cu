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
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

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

  chooseCudaDevice();
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
__host__ __device__
    void
    cartesianVertices(int idx, int *ptVec)
{
  switch (idx)
  {
    case 0:
      ptVec[0] = 0.0;
      ptVec[1] = 0.0;
      ptVec[2] = 0.0;
      break;
    case 1:
      ptVec[0] = 0.0;
      ptVec[1] = 1.0;
      ptVec[2] = 0.0;
      break;
    case 2:
      ptVec[0] = 1.0;
      ptVec[1] = 1.0;
      ptVec[2] = 0.0;
      break;
    case 3:
      ptVec[0] = 1.0;
      ptVec[1] = 0.0;
      ptVec[2] = 0.0;
      break;
  }
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

  int bz_max = DIMZ / BLOCK_SIZE;
  int cartesianVertex[3];
  int ptVec[3];

  for (bz = 0; bz < bz_max; ++bz) {
    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;
    // int z = bz * BLOCK_SIZE + tz;
    ptVec[0] = x;
    ptVec[1] = y;
    ptVec[2] = params->pos;

    x = ptVec[params->pointOrderVec[0]];
    y = ptVec[params->pointOrderVec[1]];
    int z = ptVec[params->pointOrderVec[2]];

    unsigned int idx = index_dev(x,y,z, params->dim[0], params->dim[1]);

    CellG* cell = params->cellField[idx];
    cell_types_out[idx] = (int) cell->type;

    // Coordinates3D<double> coords(ptVec[0], ptVec[1], 0);
    for (int idx = 0; idx < 4; ++idx)
    {
      cartesianVertices(idx, cartesianVertex);
      cellCoords_out[idx * 2] = cartesianVertex[0] + ptVec[0];
      cellCoords_out[(idx * 2) + 1] = cartesianVertex[1] + ptVec[1];
    }

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
  size_t cellsArraySize = numPoints * sizeof(int);
  checkCudaErrors(cudaMalloc((void**)&d_cellsArray, cellsArraySize*5));
  checkCudaErrors(cudaMallocManaged((void**)&d_cellsType, cellsArraySize));
  checkCudaErrors(cudaMallocManaged((void **)&d_cellCoords, numPoints * 2 * 4 * sizeof(double)));
  FieldExtractParams_t* params_device;
  // checkCudaErrors(cudaHostGetDevicePointer((void **) &params_device, (void *)&params, 0));
  checkCudaErrors(cudaMalloc((void**)&params_device, sizeof(FieldExtractParams_t)));
  checkCudaErrors(cudaMemcpy((void *)params_device, (void *)&params, sizeof(FieldExtractParams_t), cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(params.dim[0] / threads.x, params.dim[1] / threads.y);

  fillCellFieldData2DCartesian_gpu<<< grid, threads >>>(d_cellsArray, d_cellsType, d_cellCoords, params_device);
  vtkIdType *_cellsArrayWritePtr;
  #pragma omp parallel
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        _cellsArrayWritePtr = _cellsArray->WritePointer(numPoints, numPoints*5);
        checkCudaErrors(cudaMemcpy(_cellsArrayWritePtr, d_cellsArray, cellsArraySize, cudaMemcpyDeviceToHost));
      }
      #pragma omp section
      {
        _cellTypeArray->SetNumberOfValues(numPoints);
      }
      #pragma omp section
      {
        _pointsArray->SetNumberOfPoints(numPoints*4);
      }
      #pragma omp section
      {
        cudaDeviceSynchronize();
        cout << "post sync" << endl;
      }
    }

#pragma omp for schedule(static)
    for (int j = 0; j < numPoints; ++j) {
      int cellPos = j*4;
			for (int idx = 0; idx < 4; ++idx) {
			  // Coordinates3D<double> cartesianVertex=cartesianVertices[idx]+coords;
        _pointsArray->SetPoint(cellPos + idx, d_cellCoords[(cellPos + idx)*2], d_cellCoords[(cellPos + idx)*2 +1], 0.0);
      }
      _cellTypeArray->SetValue(j, d_cellsType[j]);
    }
  }
  cout << "post setting to VTK" << endl;

  checkCudaErrors(cudaFree(d_cellsType));
  checkCudaErrors(cudaFree(d_cellCoords));
}

void FieldExtractor_CUDA::fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos)
{
  cout << "FieldExtractor_CUDA fillBorderData2D" << endl;
  vtkPoints *points = (vtkPoints *)_pointArrayAddr;
  vtkCellArray *lines = (vtkCellArray *)_linesArrayAddr;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

  vector<int> fieldDimVec(3, 0);
  fieldDimVec[0] = fieldDim.x;
  fieldDimVec[1] = fieldDim.y;
  fieldDimVec[2] = fieldDim.z;

  int pointOrderVec[3];
  pointOrder(_plane, pointOrderVec);
  int dimOrderVec[3];
  dimOrder(_plane, dimOrderVec);

  vector<int> dim(3, 0);
  dim[0] = fieldDimVec[dimOrderVec[0]];
  dim[1] = fieldDimVec[dimOrderVec[1]];
  dim[2] = fieldDimVec[dimOrderVec[2]];

  vector<std::tuple<double, double, double, double>> global_points;
  vtkIdType *linesWritePtr;
#pragma omp parallel shared(pointOrderVec, dim, cellFieldG, points, lines, global_points, linesWritePtr)
  {
    vector<std::tuple<double, double, double, double>> local_points;
#pragma omp for schedule(static) nowait
    for (int i = 0; i < dim[0]; ++i)
    {
      Point3D pt;
      vector<int> ptVec(3, 0);
      Point3D ptN;
      vector<int> ptNVec(3, 0);

      for (int j = 0; j < dim[1]; ++j)
      {
        ptVec[0] = i;
        ptVec[1] = j;
        ptVec[2] = _pos;

        pt.x = ptVec[pointOrderVec[0]];
        pt.y = ptVec[pointOrderVec[1]];
        pt.z = ptVec[pointOrderVec[2]];

        if (i > 0 && j < dim[1])
        {
          ptNVec[0] = i - 1;
          ptNVec[1] = j;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j, (double)i, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i, j + 1));
          }
        }
        if (j > 0 && i < dim[0])
        {
          ptNVec[0] = i;
          ptNVec[1] = j - 1;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j, (double)i + 1, (double)j));
            // local_points.push_back(std::pair<double, double>(i + 1, j));
          }
        }

        if (i < dim[0] && j < dim[1])
        {
          ptNVec[0] = i + 1;
          ptNVec[1] = j;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i + 1, (double)j, (double)i + 1, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i + 1, j + 1));
          }
        }

        if (i < dim[0] && j < dim[1])
        {
          ptNVec[0] = i;
          ptNVec[1] = j + 1;
          ptNVec[2] = _pos;
          ptN.x = ptNVec[pointOrderVec[0]];
          ptN.y = ptNVec[pointOrderVec[1]];
          ptN.z = ptNVec[pointOrderVec[2]];
          if (cellFieldG->get(pt) != cellFieldG->get(ptN))
          {
            local_points.push_back(make_tuple<double, double>((double)i, (double)j + 1, (double)i + 1, (double)j + 1));
            // local_points.push_back(std::pair<double, double>(i + 1, j + 1));
          }
        }
      }
    }
#pragma omp critical
    {
      global_points.insert(global_points.end(), local_points.begin(), local_points.end());
    }

#pragma omp barrier

#pragma omp sections
    {
#pragma omp section
      {
        linesWritePtr = lines->WritePointer(global_points.size(), global_points.size() * 3);
      }
#pragma omp section
      {
        points->SetNumberOfPoints(global_points.size() * 2);
      }
    }

    int pc, pt_pos = 0;
#pragma omp for schedule(static)
    for (int j = 0; j < global_points.size(); ++j)
    {
      std::tuple<double, double, double, double> pt = global_points[j];
      pt_pos = j * 2;
      points->SetPoint(pt_pos, std::get<0>(pt), std::get<1>(pt), 0);
      points->SetPoint(pt_pos + 1, std::get<2>(pt), std::get<3>(pt), 0);
      pc = j * 3;
      linesWritePtr[pc] = 2;
      linesWritePtr[pc + 1] = pt_pos;
      linesWritePtr[pc + 2] = pt_pos + 1;
    }
  }
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

  vtkIntArray *cellTypeArray = (vtkIntArray *)_cellTypeArrayAddr;
  vtkLongArray *cellIdArray = (vtkLongArray *)_cellIdArrayAddr;

  Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
  Dim3D fieldDim = cellFieldG->getDim();

  // if neighbor tracker is loaded we can figure out cell ids that touch medium (we call them outer cells) and render only those
  // this way we do not waste time rendering inner cells that are not seen because they are covered by outer cells.
  // this algorithm is not perfect but does significantly speed up 3D rendering

  bool neighbor_tracker_loaded = Simulator::pluginManager.isLoaded("NeighborTracker");
  // cout << "neighbor_tracker_loaded=" << neighbor_tracker_loaded << endl;
  ExtraMembersGroupAccessor<NeighborTracker> *neighborTrackerAccessorPtr;
  if (neighbor_tracker_loaded)
  {
    bool pluginAlreadyRegisteredFlag;
    NeighborTrackerPlugin *nTrackerPlugin = (NeighborTrackerPlugin *)Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
    neighborTrackerAccessorPtr = nTrackerPlugin->getNeighborTrackerAccessorPtr();
  }

  std::unordered_set<long> outer_cell_ids_set;

  // to optimize drawing individual cells in 3D we may use cell shell optimization where we draw only cells that make up a cell shell opf the volume and skip inner cells that are not visible
  bool cellShellOnlyOptimization = neighbor_tracker_loaded && extractOuterShellOnly;

  if (cellShellOnlyOptimization)
  {

    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;
    CellInventory &cellInventory = potts->getCellInventory();
    // TODO: need OpenMP 3.0 > support on Windows to allow non-integer for loop indicies, cannot parallelize this
    for (cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr)
    {
      cell = cellInventory.getCell(cInvItr);
      std::set<NeighborSurfaceData> *neighborsPtr = &(neighborTrackerAccessorPtr->get(cell->extraAttribPtr)->cellNeighbors);
      set<NeighborSurfaceData>::iterator sitr;
      for (sitr = neighborsPtr->begin(); sitr != neighborsPtr->end(); ++sitr)
      {
        if (!sitr->neighborAddress)
        {
          outer_cell_ids_set.insert(cell->id);
          break;
        }
      }
    }
  }

  ParallelUtilsOpenMP *pUtils = sim->pUtils;
  // todo - consider separate CPU setting for graphics
  unsigned int num_work_nodes = pUtils->getNumberOfWorkNodes();
  vector<unordered_set<int>> vecUsedCellTypes(num_work_nodes);

#pragma omp parallel shared(vecUsedCellTypes, cellTypeArray, cellIdArray, fieldDim, outer_cell_ids_set, cellFieldG)
  {
#pragma omp sections
    {
#pragma omp section
      {
        cellTypeArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
      }
#pragma omp section
      {
        cellIdArray->SetNumberOfValues((fieldDim.x + 2) * (fieldDim.y + 2) * (fieldDim.z + 2));
      }
    }

    unsigned int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    // when accessing cell field it is OK to go outside cellfieldG limits. In this case null pointer is returned
#pragma omp for schedule(static)
    for (int k = 0; k < fieldDim.z + 2; ++k)
    {
      Point3D pt;
      CellG *cell;
      int type;
      long id;

      int k_offset = k * (fieldDim.y + 2) * (fieldDim.x + 2);
      for (int j = 0; j < fieldDim.y + 2; ++j)
      {
        int j_offset = j * (fieldDim.x + 2);
        for (int i = 0; i < fieldDim.x + 2; ++i)
        {
          int offset = k_offset + j_offset + i;
          if (i == 0 || i == fieldDim.x + 1 || j == 0 || j == fieldDim.y + 1 || k == 0 || k == fieldDim.z + 1)
          {
            cellTypeArray->SetValue(offset, 0);
            cellIdArray->SetValue(offset, 0);
          }
          else
          {
            pt.x = i - 1;
            pt.y = j - 1;
            pt.z = k - 1;
            cell = cellFieldG->get(pt);
            if (!cell)
            {
              type = 0;
              id = 0;
            }
            else
            {
              type = cell->type;
              id = cell->id;

              vecUsedCellTypes[currentWorkNodeNumber].insert(type);
              //            if (usedCellTypes.find(type) == usedCellTypes.end())
              //            {
              //              #pragma omp critical
              //              usedCellTypes.insert(type);
              //            }
            }
            if (cellShellOnlyOptimization)
            {
              if (outer_cell_ids_set.find(id) != outer_cell_ids_set.end())
              {
                cellTypeArray->SetValue(offset, type);
                cellIdArray->SetValue(offset, id);
              }
              else
              {
                cellTypeArray->SetValue(offset, 0);
                cellIdArray->SetValue(offset, 0);
              }
            }
            else
            {
              cellTypeArray->SetValue(offset, type);
              cellIdArray->SetValue(offset, id);
            }
          }
        }
      }
    }
  } // omp_parallel

  unordered_set<int> usedCellTypes;
  for (auto s : vecUsedCellTypes)
  {
    usedCellTypes.insert(s.begin(), s.end());
  }
  return vector<int>(usedCellTypes.begin(), usedCellTypes.end());
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
