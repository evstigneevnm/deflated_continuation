#ifndef __COMMON_GPU_MATRIX_FILE_OPERATIONS_WRAPPER_H__
#define __COMMON_GPU_MATRIX_FILE_OPERATIONS_WRAPPER_H__

#include <nmfd/operations/io/matrix_file_operations.h>

template<class MatrixOperations>
using gpu_matrix_file_operations = nmfd::operations::io::matrix_file_operations<MatrixOperations>;

#endif
