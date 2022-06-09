#include <iostream>
#include <CompuCell3D/Simulator.h>
#include "FieldExtractor_CUDA.cuh"

using namespace std;
using namespace CompuCell3D;

FieldExtractor_CUDA::FieldExtractor_CUDA()
{
  fsPtr = nullptr;
  potts = nullptr;
  sim = nullptr;
  cout << "constructing FieldExtractor_CUDA" << endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldExtractor_CUDA::~FieldExtractor_CUDA(){
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldExtractor_CUDA::init(Simulator *_sim)
{
  sim = _sim;
  potts = sim->getPotts();
  cout << "initing FieldExtractor_CUDA" << endl;
}
