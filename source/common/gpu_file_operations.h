#ifndef __COMMON_GPU_FILE_OPERATIONS_WRAPPER_H__
#define __COMMON_GPU_FILE_OPERATIONS_WRAPPER_H__

#include <nmfd/operations/io/vector_file_operations.h>

template<class VectorOperations>
using gpu_file_operations = nmfd::operations::io::vector_file_operations<VectorOperations>;

#endif
