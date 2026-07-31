#ifndef __NMFD_OPERATIONS_LINALG_HOST_SMALL_DENSE_LAPACK_H__
#define __NMFD_OPERATIONS_LINALG_HOST_SMALL_DENSE_LAPACK_H__

#include <complex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <scfd/external_libraries/lapack_wrap.h>

#include "host_small_dense_backend.h"

namespace nmfd
{
namespace operations
{
namespace linalg
{

template<class Real>
struct host_dense_eigensystem
{
    std::vector<std::complex<Real>> eigenvalues;
    host_dense_matrix<std::complex<Real>> right_eigenvectors;
};

template<class Real>
struct host_dense_schur_decomposition
{
    std::vector<std::complex<Real>> eigenvalues;
    host_dense_matrix<Real> orthogonal_vectors;
    host_dense_matrix<Real> quasi_triangular;
};

template<class Real>
struct host_dense_real_schur_block
{
    std::size_t first = 0;
    std::size_t size = 0;
    std::vector<std::complex<Real>> eigenvalues;
};

template<class Real>
class host_small_dense_lapack
{
    static_assert(
        std::is_same<Real, float>::value ||
        std::is_same<Real, double>::value,
        "SCFD LAPACK adapter supports float and double");

public:
    using real_type = Real;
    using complex_type = std::complex<Real>;
    using matrix_type = host_dense_matrix<Real>;

    std::vector<complex_type> eigenvalues(const matrix_type& matrix) const
    {
        check_square(matrix);
        std::vector<complex_type> result(matrix.rows());
        scfd::lapack_wrap<Real> lapack(matrix.rows());
        lapack.eigs(matrix.data(), matrix.rows(), result.data());
        return result;
    }

    std::vector<complex_type> hessenberg_eigenvalues(
        const matrix_type& matrix) const
    {
        check_square(matrix);
        std::vector<complex_type> result(matrix.rows());
        scfd::lapack_wrap<Real> lapack(matrix.rows());
        lapack.hessinberg_eigs(matrix.data(), matrix.rows(), result.data());
        return result;
    }

    host_dense_eigensystem<Real> eigensystem(const matrix_type& matrix) const
    {
        check_square(matrix);
        host_dense_eigensystem<Real> result;
        result.eigenvalues.resize(matrix.rows());
        result.right_eigenvectors.resize(matrix.rows(), matrix.cols());
        scfd::lapack_wrap<Real> lapack(matrix.rows());
        lapack.eigsv(
            matrix.data(),
            matrix.rows(),
            result.eigenvalues.data(),
            result.right_eigenvectors.data());
        return result;
    }

    host_dense_schur_decomposition<Real> hessenberg_schur(
        const matrix_type& matrix) const
    {
        check_square(matrix);
        host_dense_schur_decomposition<Real> result;
        result.eigenvalues.resize(matrix.rows());
        result.orthogonal_vectors.resize(matrix.rows(), matrix.cols());
        result.quasi_triangular.resize(matrix.rows(), matrix.cols());
        scfd::lapack_wrap<Real> lapack(matrix.rows());
        lapack.hessinberg_schur(
            matrix.data(),
            matrix.rows(),
            result.orthogonal_vectors.data(),
            result.quasi_triangular.data(),
            result.eigenvalues.data());
        return result;
    }

    host_dense_schur_decomposition<Real> schur(
        const matrix_type& matrix) const
    {
        check_square(matrix);
        host_dense_schur_decomposition<Real> result;
        result.eigenvalues.resize(matrix.rows());
        result.orthogonal_vectors.resize(matrix.rows(), matrix.cols());
        result.quasi_triangular.resize(matrix.rows(), matrix.cols());
        scfd::lapack_wrap<Real> lapack(matrix.rows());
        lapack.eigs_schur(
            matrix.data(),
            matrix.rows(),
            result.eigenvalues.data(),
            result.orthogonal_vectors.data(),
            result.quasi_triangular.data());
        return result;
    }

    std::vector<host_dense_real_schur_block<Real>> real_schur_blocks(
        const host_dense_schur_decomposition<Real>& decomposition) const
    {
        check_schur_dimensions(decomposition);
        const std::size_t dimension = decomposition.quasi_triangular.rows();
        const Real threshold =
            Real(64)*std::numeric_limits<Real>::epsilon()*
            std::max(Real(1), max_abs(decomposition.quasi_triangular));

        std::vector<host_dense_real_schur_block<Real>> result;
        for(std::size_t first = 0; first < dimension;)
        {
            host_dense_real_schur_block<Real> block;
            block.first = first;
            block.size =
                first + 1 < dimension &&
                    std::abs(
                        decomposition.quasi_triangular(first + 1, first)) >
                        threshold
                ? 2
                : 1;
            block.eigenvalues.insert(
                block.eigenvalues.end(),
                decomposition.eigenvalues.begin() +
                    static_cast<std::ptrdiff_t>(first),
                decomposition.eigenvalues.begin() +
                    static_cast<std::ptrdiff_t>(first + block.size));
            result.emplace_back(std::move(block));
            first += result.back().size;
        }
        return result;
    }

    void move_schur_block(
        host_dense_schur_decomposition<Real>& decomposition,
        std::size_t source_first,
        std::size_t destination_first) const
    {
        check_schur_dimensions(decomposition);
        const std::size_t dimension = decomposition.quasi_triangular.rows();
        if(source_first >= dimension || destination_first >= dimension)
            throw std::out_of_range("Schur block move index out of range");

        scfd::lapack_wrap<Real> lapack(dimension);
        lapack.reorder_schur(
            decomposition.quasi_triangular.data(),
            decomposition.orthogonal_vectors.data(),
            dimension,
            source_first + 1,
            destination_first + 1);
        refresh_schur_eigenvalues(decomposition);
    }

    void refresh_schur_eigenvalues(
        host_dense_schur_decomposition<Real>& decomposition) const
    {
        check_schur_dimensions(decomposition);
        const std::size_t dimension = decomposition.quasi_triangular.rows();
        decomposition.eigenvalues.resize(dimension);
        scfd::lapack_wrap<Real> lapack(dimension);
        lapack.schur_upper_triag_ordered_eigs(
            decomposition.quasi_triangular.data(),
            dimension,
            decomposition.eigenvalues.data());
    }

private:
    static void check_square(const matrix_type& matrix)
    {
        if(matrix.rows() == 0 || matrix.rows() != matrix.cols())
            throw std::invalid_argument(
                "host_small_dense_lapack requires a nonempty square matrix");
    }

    static void check_schur_dimensions(
        const host_dense_schur_decomposition<Real>& decomposition)
    {
        check_square(decomposition.quasi_triangular);
        const std::size_t dimension = decomposition.quasi_triangular.rows();
        if(
            decomposition.orthogonal_vectors.rows() != dimension ||
            decomposition.orthogonal_vectors.cols() != dimension ||
            decomposition.eigenvalues.size() != dimension)
        {
            throw std::invalid_argument(
                "host_small_dense_lapack Schur decomposition size mismatch");
        }
    }

    static Real max_abs(const matrix_type& matrix)
    {
        Real result = Real{};
        for(std::size_t col = 0; col < matrix.cols(); ++col)
            for(std::size_t row = 0; row < matrix.rows(); ++row)
                result = std::max(result, std::abs(matrix(row, col)));
        return result;
    }
};

} // namespace linalg
} // namespace operations
} // namespace nmfd

#endif
