#ifndef __CPU_MATRIX_FILE_OPERATIONS_H__
#define __CPU_MATRIX_FILE_OPERATIONS_H__

#include <common/file_operations.h>

template<class MatrixOperations>
class cpu_matrix_file_operations
{
public:
    typedef typename MatrixOperations::scalar_type  T;
    typedef typename MatrixOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type  T_mat;

    cpu_matrix_file_operations(MatrixOperations* mat_op_):
    mat_op(mat_op_)
    {
        sz_row = mat_op->get_rows();
        sz_col = mat_op->get_cols();      
    }

    ~cpu_matrix_file_operations()
    {

    }

    std::pair<size_t, size_t> read_matrix_size(const std::string &f_name)
    {
        return file_operations::read_matrix_size(f_name);
    }

    void write_matrix(const std::string f_name, T_mat& mat, unsigned int prec=16) const
    {
        file_operations::write_matrix<T_mat>(f_name, sz_row, sz_col, mat, prec);
    }

    void read_matrix(const std::string &f_name, T_mat& mat) const
    {
        file_operations::read_matrix<T, T_mat>(f_name, sz_row, sz_col, mat);
    }

    
private:
    MatrixOperations* mat_op;
    size_t sz_row;
    size_t sz_col;
};




#endif // __CPU_FILE_OPERATIONS_H__