#ifndef FIELDEXTRACTOR_CUDA
#define FIELDEXTRACTOR_CUDA

#include <iostream>
#include "FieldExtractorBase.h"
#include "FieldStorage.h"
#include "FieldExtractorDLLSpecifier.h"
// #include <CompuCell3D/Potts3D/Cell.h>

namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class Potts3D;
	class Simulator;
	class Dim3D;
  class NeighborTracker;

	class FIELDEXTRACTOR_EXPORT FieldExtractor_CUDA : public FieldExtractorBase{
      public:
        Potts3D *potts;
        Simulator *sim;

        FieldExtractor_CUDA();
        ~FieldExtractor_CUDA();

        void init(Simulator *_sim);

      private:
        FieldStorage *fsPtr;
  };
};
#endif // FIELDEXTRACTOR_CUDA