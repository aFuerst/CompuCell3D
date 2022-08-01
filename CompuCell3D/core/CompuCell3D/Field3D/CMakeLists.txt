SET(SRCS
  AdjacentNeighbor.cpp
  Field3DChangeWatcher.cpp
  Field3DIndex.cpp
  Point3D.cpp
  Dim3D.cpp
  Field3D.cpp
  # Field3DImpl.cu
  # Field3DImpl.cuh
  NeighborFinder.cpp
  WatchableField3D.cpp
)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAToolkit)
INCLUDE(FindCUDA)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_60
    )
    
# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#    CUDA_ADD_LIBRARY(Field3D STATIC ${SRCS})
# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#    CUDA_ADD_LIBRARY(Field3D SHARED ${SRCS})
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES cuda ${CUDA_LIBRARIES})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES cuda ${CUDA_LIBRARIES})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

set_target_properties(${Field3DLib} PROPERTIES CUDA_ARCHITECTURES "60")

TARGET_INCLUDE_DIRECTORIES(${Field3DLib} PUBLIC
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
  ${Python_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  /extra/alfuerst/cc3d_2022/include/
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D/Field3D FILES
  AdjacentNeighbor.h
  Dim3D.h
  Array3D.h
  Field3DChangeWatcher.h
  Field3D.h
  Field3DImpl.cuh
  Field3DIndex.h
  Field3DIO.h
  NeighborFinder.h
  Neighbor.h
  Point3D.h
  WatchableField3D.h
  CC3D_Field3D.h
)
