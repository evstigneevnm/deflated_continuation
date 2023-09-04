#ifndef __GPU_MATRIX_FILE_OPERATIONS_H__
#define __GPU_MATRIX_FILE_OPERATIONS_H__

#include <common/gpu_file_operations_functions.h>

template<class MatrixOperations>
class gpu_matrix_file_operations
{
public:
    typedef typename MatrixOperations::scalar_type  T;
    typedef typename MatrixOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type  T_mat;

    gpu_matrix_file_operations(MatrixOperations* mat_op_):
    mat_op(mat_op_)
    {
        sz_row = mat_op->get_rows();
        sz_col = mat_op->get_cols();      
    }

    ~gpu_matrix_file_operations()
    {

    }

    void write_matrix(const std::string f_name, T_mat& mat_gpu, unsigned int prec=16) const
    {
        gpu_file_operations_functions::write_matrix<T>(f_name, sz_row, sz_col, mat_gpu, prec);
    }

    void read_matrix(const std::string &f_name, T_mat& mat_gpu) const
    {
        
        gpu_file_operations_functions::read_matrix<T>(f_name, sz_row, sz_col, mat_gpu);
    }

    
private:
    MatrixOperations* mat_op;
    size_t sz_row;
    size_t sz_col;
};




#endif // __GPU_FILE_OPERATIONS_H__