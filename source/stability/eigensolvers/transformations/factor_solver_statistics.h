#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_FACTOR_SOLVER_STATISTICS_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_FACTOR_SOLVER_STATISTICS_H__

#include <cstddef>
#include <limits>
#include <vector>

#include "complex_affine_factor.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class Real, class Norm = Real>
struct factor_solver_statistics
{
    std::size_t index = 0;
    complex_affine_factor<Real> descriptor{};
    std::size_t solve_calls = 0;
    std::size_t failed_solves = 0;
    std::size_t total_iterations = 0;
    std::size_t maximum_iterations = 0;
    std::size_t last_iterations = 0;
    Norm last_residual = Norm{};
    bool last_residual_available = false;
};

template<class Real, class Norm = Real>
struct factor_solver_bundle_statistics
{
    static constexpr std::size_t no_factor_index =
        std::numeric_limits<std::size_t>::max();

    std::size_t solve_calls = 0;
    std::size_t factor_solve_calls = 0;
    std::size_t failed_solves = 0;
    std::size_t last_failed_factor = no_factor_index;
    std::size_t completed_factors_last_solve = 0;
    std::vector<factor_solver_statistics<Real, Norm>> factors;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
