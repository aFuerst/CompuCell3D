#ifndef FIELDEXTRACTOR_CUDA
#define FIELDEXTRACTOR_CUDA

#include <iostream>
#include "FieldExtractorBase.h"
#include "FieldExtractor.h"
#include "FieldStorage.h"
#include "FieldExtractorDLLSpecifier.h"
// #include <CompuCell3D/Potts3D/Cell.h>

namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class Potts3D;
	class Simulator;
	class Dim3D;
  class NeighborTracker;

  class FIELDEXTRACTOR_EXPORT FieldExtractor_CUDA : public FieldExtractor // FieldExtractorBase
  {
  public:
    Potts3D *potts;
    Simulator *sim;

    FieldExtractor_CUDA();
    ~FieldExtractor_CUDA();

    virtual void setFieldStorage(FieldStorage *_fsPtr);
    virtual FieldStorage *getFieldStorage(FieldStorage *_fsPtr) { return fsPtr; }

    // void extractCellField();

    virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _plane, int _pos);
    virtual void fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);
    virtual void fillCellFieldData2DHex_old(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);
    virtual void fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);

    virtual void fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos);
    virtual void fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos);

    virtual void fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos);
    virtual void fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos);

    virtual void fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr, vtk_obj_addr_int_t _linesArrayAddr, std::string _plane, int _pos);

    virtual bool fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos);
    virtual bool fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);

    virtual bool fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);

    virtual bool fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos);
    virtual bool fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);
    virtual bool fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);

    virtual bool fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr, std::string _conFieldName, std::string _plane, int _pos);
    virtual bool fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);
    virtual bool fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _conFieldName, std::string _plane, int _pos);

    virtual bool fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec);

    virtual bool fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos);
    virtual bool fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos);
    virtual bool fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName);
    virtual bool fillVectorFieldData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName);

    virtual bool fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos);
    virtual bool fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName, std::string _plane, int _pos);
    virtual bool fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName);
    virtual bool fillVectorFieldCellLevelData3DHex(vtk_obj_addr_int_t _pointsArrayIntAddr, vtk_obj_addr_int_t _vectorArrayIntAddr, std::string _fieldName);

    virtual bool fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec);

    virtual std::vector<int> fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr, bool extractOuterShellOnly = false);
    virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr, vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName, std::vector<int> *_typesInvisibeVec);

    void init(Simulator *_sim);

  private:
    FieldStorage *fsPtr;
    void alloc();

    float* moveCellFieldToDevice(std::string _conFieldName);
    float* moveConFieldToDevice(std::string _conFieldName);

    // device
    float *cell_field;
    float *con_field;
  };
};
#endif // FIELDEXTRACTOR_CUDA