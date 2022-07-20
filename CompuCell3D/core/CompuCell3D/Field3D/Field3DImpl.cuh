#ifndef FIELD3DIMPL_H
#define FIELD3DIMPL_H

#include <math.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/CC3DExceptions.h>

#include "Dim3D.h"
#include "Field3D.h"
// #include <CompuCell3D/steppables/PDESolvers/CUDA/CUDAUtilsHeader.h>
#include <CompuCell3D/CudaUtils.cuh>
#include <cuda.h>

namespace CompuCell3D {
//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * dim.y)) * dim.x))

template<class T>
  class Field3D;

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
    Field3DImpl(const Dim3D dim, const T &initialValue);

    ~Field3DImpl();

    T getQuick(const Point3D &pt) const;

    void setQuick(const Point3D &pt, const T _value);

    virtual T* getPtr() override;
    virtual void set(const Point3D &pt, const T value) override;
    virtual T get(const Point3D &pt) const override;
    virtual T getByIndex(long _offset) const override;
    virtual void setByIndex(long _offset, const T value) override;
    virtual T operator[](const Point3D &pt) const override;
    virtual Dim3D getDim() const override;
    virtual bool isValid(const Point3D &pt) const override;
    virtual void setDim(const Dim3D theDim) override;
    virtual void resizeAndShift(const Dim3D theDim, const Dim3D shiftVec) override;
    virtual void clearSecData() override;
  };
};
#endif
