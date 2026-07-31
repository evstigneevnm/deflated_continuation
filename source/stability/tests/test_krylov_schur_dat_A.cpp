#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/eigensolvers/krylov_schur.h>

namespace
{

struct dense_matrix
{
    std::size_t dimension = 0;
    std::vector<double> row_major;
    double frobenius_norm = 0.0;
};

dense_matrix read_dense_square_matrix(const std::string& file_name)
{
    std::ifstream file(file_name);
    if(!file)
        throw std::runtime_error("failed to open matrix " + file_name);

    dense_matrix result;
    std::string line;
    std::size_t columns = 0;
    while(std::getline(file, line))
    {
        if(line.empty())
            continue;
        std::istringstream values(line);
        std::vector<double> row;
        double value = 0.0;
        while(values >> value)
        {
            row.push_back(value);
            result.frobenius_norm += value*value;
        }
        if(row.empty())
            continue;
        if(columns == 0)
            columns = row.size();
        if(row.size() != columns)
            throw std::runtime_error("inconsistent matrix row length");
        result.row_major.insert(
            result.row_major.end(),
            row.begin(),
            row.end());
        ++result.dimension;
    }
    if(
        result.dimension == 0 ||
        columns != result.dimension ||
        result.row_major.size() != result.dimension*result.dimension)
    {
        throw std::runtime_error("matrix is not nonempty and square");
    }
    result.frobenius_norm = std::sqrt(result.frobenius_norm);
    return result;
}

std::vector<std::complex<double>> read_reference(
    const std::string& file_name)
{
    std::ifstream file(file_name);
    if(!file)
        throw std::runtime_error("failed to open reference " + file_name);

    std::vector<std::complex<double>> result;
    std::string line;
    std::getline(file, line);
    while(std::getline(file, line))
    {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream values(line);
        std::size_t rank = 0;
        double real = 0.0;
        double imaginary = 0.0;
        if(values >> rank >> real >> imaginary)
            result.emplace_back(real, imaginary);
    }
    if(result.empty())
        throw std::runtime_error("empty eigenvalue reference");
    return result;
}

template<class VectorSpace>
class host_dense_reference_operator
{
public:
    using vector_type = typename VectorSpace::vector_type;

    host_dense_reference_operator(
        const VectorSpace& vector_space,
        const dense_matrix& matrix)
        : vector_space_(vector_space),
          matrix_(matrix),
          source_(matrix.dimension),
          destination_(matrix.dimension)
    {
    }

    bool apply(const vector_type& source, vector_type& destination)
    {
        ++operator_calls_;
        vector_space_.get(source, source_.data(), source_.size());

#pragma omp parallel for schedule(static)
        for(std::ptrdiff_t row = 0;
            row < static_cast<std::ptrdiff_t>(matrix_.dimension);
            ++row)
        {
            double value = 0.0;
            const std::size_t offset =
                static_cast<std::size_t>(row)*matrix_.dimension;
            for(std::size_t col = 0; col < matrix_.dimension; ++col)
                value += matrix_.row_major[offset + col]*source_[col];
            destination_[static_cast<std::size_t>(row)] = value;
        }
        vector_space_.set(
            destination_.data(),
            destination,
            destination_.size());
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

private:
    const VectorSpace& vector_space_;
    const dense_matrix& matrix_;
    std::vector<double> source_;
    std::vector<double> destination_;
    std::size_t operator_calls_ = 0;
};

std::vector<std::size_t> match_reference(
    const std::vector<std::complex<double>>& reference,
    const std::vector<
        stability::eigensolvers::eigenpair_estimate<double>>& computed)
{
    std::vector<std::size_t> result(reference.size(), computed.size());
    std::vector<bool> used(computed.size(), false);
    for(std::size_t expected = 0; expected < reference.size(); ++expected)
    {
        double best_error = std::numeric_limits<double>::infinity();
        for(std::size_t candidate = 0; candidate < computed.size(); ++candidate)
        {
            if(used[candidate])
                continue;
            const double error =
                std::abs(reference[expected] - computed[candidate].value);
            if(error < best_error)
            {
                best_error = error;
                result[expected] = candidate;
            }
        }
        if(result[expected] < computed.size())
            used[result[expected]] = true;
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string matrix_file =
            argc > 1 ? argv[1] : "dat_files/A.dat";
        const std::string reference_file =
            argc > 2
            ? argv[2]
            : "data/reference/eigensolvers/baseline_20260725/"
              "dat_A_lr_reference.csv";

        const dense_matrix matrix =
            read_dense_square_matrix(matrix_file);
        const auto reference = read_reference(reference_file);
        using vector_space_type =
            scfd_vector_operations<scfd::backend::omp, double>;
        using vector_type = typename vector_space_type::vector_type;
        using lapack_type =
            nmfd::operations::linalg::host_small_dense_lapack<double>;
        using solver_type =
            stability::eigensolvers::krylov_schur<
                vector_space_type,
                lapack_type>;

        vector_space_type vector_space(matrix.dimension);
        host_dense_reference_operator<vector_space_type> matrix_operator(
            vector_space,
            matrix);
        lapack_type lapack;
        solver_type solver(vector_space, lapack);

        vector_type initial;
        vector_type real_vector;
        vector_type imaginary_vector;
        vector_type applied_real;
        vector_type applied_imaginary;
        vector_type residual_real;
        vector_type residual_imaginary;
        vector_space.init_vector(initial);
        vector_space.init_vector(real_vector);
        vector_space.init_vector(imaginary_vector);
        vector_space.init_vector(applied_real);
        vector_space.init_vector(applied_imaginary);
        vector_space.init_vector(residual_real);
        vector_space.init_vector(residual_imaginary);
        vector_space.start_use_vector(initial);
        vector_space.start_use_vector(real_vector);
        vector_space.start_use_vector(imaginary_vector);
        vector_space.start_use_vector(applied_real);
        vector_space.start_use_vector(applied_imaginary);
        vector_space.start_use_vector(residual_real);
        vector_space.start_use_vector(residual_imaginary);

        std::vector<double> initial_host(matrix.dimension);
        for(std::size_t index = 0; index < matrix.dimension; ++index)
        {
            initial_host[index] =
                std::sin(0.13*static_cast<double>(index + 1)) +
                0.5*std::cos(0.037*static_cast<double>(index + 1));
        }
        vector_space.set(
            initial_host.data(),
            initial,
            initial_host.size());

        typename solver_type::options_type options;
        options.desired_eigenvalues = reference.size();
        options.krylov_dimension = 48;
        options.restart_dimension = 28;
        options.max_restarts = 200;
        options.absolute_tolerance = 1e-9;
        options.relative_tolerance = 1e-10;
        options.target.kind =
            stability::eigensolvers::spectrum_target::largest_real;
        options.orthogonalization.method =
            nmfd::solvers::krylov::orthogonalization_method::
                modified_gram_schmidt;
        options.orthogonalization.reorthogonalization =
            nmfd::solvers::krylov::reorthogonalization_policy::dgks;
        options.orthogonalization.max_passes = 2;

        stability::eigensolvers::ritz_vector_storage<vector_space_type>
            recovered(vector_space, reference.size() + 1);
        auto apply = [&matrix_operator](
                         const vector_type& source,
                         vector_type& destination)
        {
            return matrix_operator.apply(source, destination);
        };
        const auto result =
            solver.execute(apply, initial, options, &recovered);
        const std::size_t solver_operator_calls = result.operator_calls;

        std::vector<double> physical_residuals(result.eigenpairs.size());
        for(std::size_t index = 0; index < result.eigenpairs.size(); ++index)
        {
            vector_space.assign(
                recovered.real(),
                static_cast<typename vector_space_type::ordinal_type>(
                    recovered.capacity()),
                static_cast<typename vector_space_type::ordinal_type>(index),
                real_vector);
            vector_space.assign(
                recovered.imaginary(),
                static_cast<typename vector_space_type::ordinal_type>(
                    recovered.capacity()),
                static_cast<typename vector_space_type::ordinal_type>(index),
                imaginary_vector);
            matrix_operator.apply(real_vector, applied_real);
            matrix_operator.apply(imaginary_vector, applied_imaginary);

            const double real = result.eigenpairs[index].value.real();
            const double imaginary = result.eigenpairs[index].value.imag();
            vector_space.assign(applied_real, residual_real);
            vector_space.add_lin_comb(
                -real,
                real_vector,
                1.0,
                residual_real);
            vector_space.add_lin_comb(
                imaginary,
                imaginary_vector,
                1.0,
                residual_real);
            vector_space.assign(applied_imaginary, residual_imaginary);
            vector_space.add_lin_comb(
                -imaginary,
                real_vector,
                1.0,
                residual_imaginary);
            vector_space.add_lin_comb(
                -real,
                imaginary_vector,
                1.0,
                residual_imaginary);

            const double residual = std::sqrt(
                vector_space.norm_sq(residual_real) +
                vector_space.norm_sq(residual_imaginary));
            const double vector_norm = std::sqrt(
                vector_space.norm_sq(real_vector) +
                vector_space.norm_sq(imaginary_vector));
            physical_residuals[index] =
                residual/
                ((matrix.frobenius_norm +
                  std::abs(result.eigenpairs[index].value))*vector_norm);
        }

        const auto assignment =
            match_reference(reference, result.eigenpairs);
        double maximum_eigenvalue_error = 0.0;
        double maximum_physical_residual = 0.0;
        std::cout << std::setprecision(17);
        std::cout
            << "rank,computed_real,computed_imag,reference_real,"
               "reference_imag,absolute_error,ritz_residual,"
               "physical_relative_residual\n";
        for(std::size_t rank = 0; rank < reference.size(); ++rank)
        {
            if(assignment[rank] >= result.eigenpairs.size())
                throw std::runtime_error("unable to match computed spectrum");
            const std::size_t computed = assignment[rank];
            const double error =
                std::abs(
                    result.eigenpairs[computed].value -
                    reference[rank]);
            maximum_eigenvalue_error =
                std::max(maximum_eigenvalue_error, error);
            maximum_physical_residual =
                std::max(
                    maximum_physical_residual,
                    physical_residuals[computed]);
            std::cout
                << rank << ','
                << result.eigenpairs[computed].value.real() << ','
                << result.eigenpairs[computed].value.imag() << ','
                << reference[rank].real() << ','
                << reference[rank].imag() << ','
                << error << ','
                << result.eigenpairs[computed].residual << ','
                << physical_residuals[computed] << '\n';
        }
        std::cout
            << "status="
            << stability::eigensolvers::eigensolver_status_name(result.status)
            << " iterations=" << result.iterations
            << " restarts=" << result.restarts
            << " solver_operator_calls=" << solver_operator_calls
            << " validation_operator_calls="
            << matrix_operator.operator_calls() - solver_operator_calls
            << " max_eigenvalue_error=" << maximum_eigenvalue_error
            << " max_physical_relative_residual="
            << maximum_physical_residual << '\n';

        vector_space.stop_use_vector(residual_imaginary);
        vector_space.stop_use_vector(residual_real);
        vector_space.stop_use_vector(applied_imaginary);
        vector_space.stop_use_vector(applied_real);
        vector_space.stop_use_vector(imaginary_vector);
        vector_space.stop_use_vector(real_vector);
        vector_space.stop_use_vector(initial);
        vector_space.free_vector(residual_imaginary);
        vector_space.free_vector(residual_real);
        vector_space.free_vector(applied_imaginary);
        vector_space.free_vector(applied_real);
        vector_space.free_vector(imaginary_vector);
        vector_space.free_vector(real_vector);
        vector_space.free_vector(initial);

        const bool passed =
            result.status ==
                stability::eigensolvers::eigensolver_status::success &&
            maximum_eigenvalue_error <= 1e-6 &&
            maximum_physical_residual <= 1e-8;
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
