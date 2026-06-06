#ifndef __NMFD_OPERATIONS_IO_MATRIX_FILE_OPERATIONS_H__
#define __NMFD_OPERATIONS_IO_MATRIX_FILE_OPERATIONS_H__

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <nmfd/operations/io/file_operations.h>

namespace nmfd
{
namespace operations
{
namespace io
{

namespace detail
{

template<class MatrixOperations, class Matrix>
auto matrix_view(int, MatrixOperations* mat_ops, Matrix& matrix) -> decltype(mat_ops->view(matrix))
{
    return mat_ops->view(matrix);
}

template<class MatrixOperations, class Matrix>
Matrix& matrix_view(long, MatrixOperations*, Matrix& matrix)
{
    static_assert(!std::is_pointer<Matrix>::value, "matrix_file_operations requires MatrixOperations::view() for pointer matrix types");
    return matrix;
}

template<class MatrixOperations, class Matrix>
auto const_matrix_view(int, MatrixOperations* mat_ops, const Matrix& matrix) -> decltype(mat_ops->view(matrix))
{
    return mat_ops->view(matrix);
}

template<class MatrixOperations, class Matrix>
const Matrix& const_matrix_view(long, MatrixOperations*, const Matrix& matrix)
{
    static_assert(!std::is_pointer<Matrix>::value, "matrix_file_operations requires MatrixOperations::view() for pointer matrix types");
    return matrix;
}

template<class MatrixOperations, class Matrix>
auto sync_matrix(int, MatrixOperations* mat_ops, Matrix& matrix) -> decltype(mat_ops->set(matrix), void())
{
    mat_ops->set(matrix);
}

template<class MatrixOperations, class Matrix>
void sync_matrix(long, MatrixOperations*, Matrix&)
{
}

} // namespace detail

template<class MatrixOperations>
class matrix_file_operations
{
public:
    using scalar_type = typename MatrixOperations::scalar_type;
    using vector_type = typename MatrixOperations::vector_type;
    using matrix_type = typename MatrixOperations::matrix_type;

    explicit matrix_file_operations(MatrixOperations* mat_ops_):
        mat_ops(mat_ops_),
        sz_row(mat_ops_->get_rows()),
        sz_col(mat_ops_->get_cols())
    {
    }

    std::pair<std::size_t, std::size_t> read_matrix_size(const std::string& f_name) const
    {
        return file_operations::read_matrix_size(f_name);
    }

    void write_matrix(const std::string& f_name, const matrix_type& matrix, unsigned int prec = 16) const
    {
        decltype(auto) host_view = detail::const_matrix_view(0, mat_ops, matrix);
        file_operations::write_matrix(f_name, sz_row, sz_col, host_view, prec);
    }

    void read_matrix(const std::string& f_name, matrix_type& matrix) const
    {
        decltype(auto) host_view = detail::matrix_view(0, mat_ops, matrix);
        file_operations::read_matrix<scalar_type>(f_name, sz_row, sz_col, host_view);
        detail::sync_matrix(0, mat_ops, matrix);
    }

private:
    MatrixOperations* mat_ops;
    std::size_t sz_row;
    std::size_t sz_col;
};

} // namespace io
} // namespace operations
} // namespace nmfd

#endif
