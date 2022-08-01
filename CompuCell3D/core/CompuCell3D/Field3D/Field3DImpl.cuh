#ifndef FIELD3DIMPL_CUH
#define FIELD3DIMPL_CUH

#include <math.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/CC3DExceptions.h>

#include "Dim3D.h"
#include "Field3D.h"
#include "Point3D.h"
// #include <CompuCell3D/steppables/PDESolvers/CUDA/CUDAUtilsHeader.h>
#include <CompuCell3D/CudaUtils.cuh>
#include <cuda_runtime.h>

namespace CompuCell3D {
//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * dim.y)) * dim.x))

template<class T>
  class Field3D;
  // class BoundaryStrategy;

  /**
   * Default implementation of the Field3D interface.
   *
   */
  template<class T>
  class Field3DImpl : public Field3D<T> {
  protected:
      Dim3D dim;
      T *field;
      T initialValue;
      long len;
  public:

      /**
       * @param dim The field dimensions
       * @param initialValue The initial value of all data elements in the field.
       */
    Field3DImpl(const Dim3D dim, const T &initialValue): dim(dim), field(0), initialValue(initialValue) {
      if (dim.x == 0 && dim.y == 0 && dim.z == 0)
        throw CC3DException("Field3D cannot have a 0 dimension!!!");

      // Check that the dimensions are not too large.
      if (log((double)dim.x) / log(2.0) + log((double)dim.y) / log(2.0) + log((double)dim.z) / log(2.0) >
          sizeof(int) * 8)
        throw CC3DException("Field3D dimensions too large!!!");

      // Allocate and initialize the field
      len = dim.x * dim.y * dim.z;
      // field = new T[len];
      checkCudaErrors(cudaMallocManaged(&field, len * sizeof(T)));
      for (unsigned int i = 0; i < len; i++)
        field[i] = initialValue;
    }

    ~Field3DImpl() {
      if (field)
      {
        // delete[] field;
        checkCudaErrors(cudaFree(field));
        field = 0;
      }
    }

    T getQuick(const Point3D &pt) const {
      // return field[PT2IDX(pt)];
      return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
    }

    void setQuick(const Point3D &pt, const T _value) {
      field[PT2IDX(pt)] = _value;
    }

    virtual T* getPtr() override{
      return field;
    }
    virtual void set(const Point3D &pt, const T value) override{
      if (!isValid(pt))
        throw CC3DException("set() point out of range!");
      field[PT2IDX(pt)] = value;
    }
    virtual T get(const Point3D &pt) const override{
      return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
    }
    virtual T getByIndex(long _offset) const override{
      return (((0 <= _offset) && (_offset < len)) ? field[_offset] : initialValue);
    }
    virtual void setByIndex(long _offset, const T _value) override{
      if ((0 <= _offset) && (_offset < len))
        field[_offset] = _value;
    }
    virtual T operator[](const Point3D &pt) const override  { return get(pt); }
    virtual Dim3D getDim() const override {
      return dim;
    }
    virtual bool isValid(const Point3D &pt) const override{
      return (0 <= pt.x && pt.x < dim.x &&
              0 <= pt.y && pt.y < dim.y &&
              0 <= pt.z && pt.z < dim.z);
    }

    virtual void setDim(const Dim3D theDim) override{
      if (dim.x == 0 && dim.y == 0 && dim.z == 0)
        throw CC3DException("Field3D cannot have a 0 dimension!!!");

      this->resizeAndShift(theDim);
    }
    virtual void resizeAndShift(const Dim3D theDim, const Dim3D shiftVec = Dim3D()) override{
      // T *field2 = new T[theDim.x * theDim.y * theDim.z];
      T *field2;
      checkCudaErrors(cudaMallocManaged(&field2, theDim.x * theDim.y * theDim.z * sizeof(T)));

      // first initialize the lattice with initial value
      for (long int i = 0; i < theDim.x * theDim.y * theDim.z; ++i)
        field2[i] = initialValue;

      // then  copy old field
      for (int x = 0; x < theDim.x; x++)
        for (int y = 0; y < theDim.y; y++)
          for (int z = 0; z < theDim.z; z++)
            if ((x - shiftVec.x >= 0) && (x - shiftVec.x < dim.x) && (y - shiftVec.y >= 0) &&
                (y - shiftVec.y < dim.y) && (z - shiftVec.z >= 0) && (z - shiftVec.z < dim.z))
            {
              field2[x + ((y + (z * theDim.y)) * theDim.x)] = getQuick(
                  Point3D(x - shiftVec.x, y - shiftVec.y, z - shiftVec.z));
            }

      // delete[]  field;
      checkCudaErrors(cudaFree(field));
      field = field2;
      dim = theDim;

      // Set dimension for the Boundary Strategy
      BoundaryStrategy::getInstance()->setDim(dim);
    }
  };
};
#endif
