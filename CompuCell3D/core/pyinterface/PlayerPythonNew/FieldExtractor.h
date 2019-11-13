#ifndef FIELDEXTRACTOR_H
#define FIELDEXTRACTOR_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "FieldStorage.h"

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/plugins/ECMaterials/ECMaterialsPlugin.h>

#include "FieldExtractorBase.h"

#include "FieldExtractorDLLSpecifier.h"

class FieldStorage;
class vtkIntArray;
class vtkDoubleArray;
class vtkFloatArray;
class vtkPoints;
class vtkCellArray;
class vtkObject;




//Notice one can speed up filling up of the Hex lattice data by allocating e.g. hexPOints ot cellType arrays
//instead of inserting values. Inserting causes reallocations and this slows down the task completion

namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class Potts3D;
	class Simulator;
	class Dim3D;
    class NeighborTracker;
	class ECMaterialsPlugin;

	class FIELDEXTRACTOR_EXPORT FieldExtractor:public FieldExtractorBase{
    private:

        
	public:
		Potts3D * potts;
		Simulator *sim;
		ECMaterialsPlugin *ecmPlugin;
		FieldExtractor();
		~FieldExtractor();

		void setFieldStorage(FieldStorage * _fsPtr){fsPtr=_fsPtr;}
		FieldStorage * getFieldStorage(FieldStorage * _fsPtr){return fsPtr;}

		void extractCellField();

		virtual void fillCellFieldData2D(vtk_obj_addr_int_t _cellTypeArrayAddr , std::string _plane ,  int _pos);
        virtual void fillCellFieldData2DCartesian(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos);
		virtual void fillCellFieldData2DHex_old(vtk_obj_addr_int_t _cellTypeArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane ,  int _pos);
	    virtual void fillCellFieldData2DHex(vtk_obj_addr_int_t _cellTypeArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane ,  int _pos);

		virtual void fillBorderData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos);
		virtual void fillBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos);

		virtual void fillClusterBorderData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos);
		virtual void fillClusterBorderData2DHex(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos);

		virtual void fillCentroidData2D(vtk_obj_addr_int_t _pointArrayAddr ,vtk_obj_addr_int_t _linesArrayAddr, std::string _plane ,  int _pos);

		virtual bool fillConFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		virtual bool fillConFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // virtual bool fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        // {return false;}
        virtual bool fillConFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

	    virtual bool fillScalarFieldData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
	    virtual bool fillScalarFieldData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        virtual bool fillScalarFieldData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);

		virtual bool fillScalarFieldCellLevelData2D(vtk_obj_addr_int_t _conArrayAddr,std::string _conFieldName, std::string _plane ,  int _pos);
		virtual bool fillScalarFieldCellLevelData2DHex(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _hexCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
        virtual bool fillScalarFieldCellLevelData2DCartesian(vtk_obj_addr_int_t _conArrayAddr,vtk_obj_addr_int_t _cartesianCellsArrayAddr ,vtk_obj_addr_int_t _pointsArrayAddr , std::string _conFieldName , std::string _plane ,int _pos);
		virtual bool fillScalarFieldCellLevelData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		virtual bool fillVectorFieldData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName);

		virtual bool fillVectorFieldCellLevelData2D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldCellLevelData2DHex(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName, std::string _plane ,  int _pos);
		virtual bool fillVectorFieldCellLevelData3D(vtk_obj_addr_int_t _pointsArrayIntAddr,vtk_obj_addr_int_t _vectorArrayIntAddr,std::string _fieldName);

	    virtual bool fillScalarFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		virtual std::vector<int> fillCellFieldData3D(vtk_obj_addr_int_t _cellTypeArrayAddr, vtk_obj_addr_int_t _cellIdArrayAddr);
		virtual bool fillConFieldData3D(vtk_obj_addr_int_t _conArrayAddr ,vtk_obj_addr_int_t _cellTypeArrayAddr, std::string _conFieldName,std::vector<int> * _typesInvisibeVec);

		void initECMaterials(ECMaterialsPlugin *_ecmPlugin);
		void extractECMaterialField();
		virtual void fillECMaterialFieldData2D(vtk_obj_addr_int_t _ecmQuantityArrayAddr, std::string _plane, int _pos, int _compSel);
		virtual void fillECMaterialFieldData2DHex(vtk_obj_addr_int_t _ecmQuantityArrayAddr, vtk_obj_addr_int_t _hexCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel);
		virtual void fillECMaterialData2DCartesian(vtk_obj_addr_int_t _ecmQuantityArrayAddr, vtk_obj_addr_int_t _cartesianCellsArrayAddr, vtk_obj_addr_int_t _pointsArrayAddr, std::string _plane, int _pos, int _compSel);
		virtual void fillECMaterialFieldData3D(vtk_obj_addr_int_t _ecmQuantityArrayAddr, int _compSel);
		void fillECMaterialDisplayField(vtk_obj_addr_int_t _colorsArrayAddr, vtk_obj_addr_int_t _quantityArrayAddr, vtk_obj_addr_int_t _colors_lutAddr);

		void setVtkObj(void * _vtkObj);
		void setVtkObjInt(long _vtkObjAddr);
		vtkIntArray * produceVtkIntArray();

		int * produceArray(int _size);
		void init(Simulator * _sim);
	private:
		FieldStorage * fsPtr;
	};
};

#endif
