#ifndef __SYMMETRY_SLICE_PROJECTOR_H__
#define __SYMMETRY_SLICE_PROJECTOR_H__

#include <cstddef>

#include <nmfd/operations/linalg/small_dense.h>

namespace symmetry
{

template <class Real, std::size_t MaxRank>
struct slice_projection_result
{
    nmfd::operations::linalg::small_solve_status status      = nmfd::operations::linalg::small_solve_status::success;
    std::size_t                                  active_rank = 0;
    nmfd::operations::linalg::small_vector<Real, MaxRank> alpha;
    nmfd::operations::linalg::small_solve_info<Real>      solve_info;

    bool ok() const
    {
        return status == nmfd::operations::linalg::small_solve_status::success;
    }
};

template <class Real, std::size_t MaxRank>
slice_projection_result<Real, MaxRank> solve_slice_projection(
    const nmfd::operations::linalg::small_matrix<Real, MaxRank, MaxRank> &slice_matrix,
    const nmfd::operations::linalg::small_vector<Real, MaxRank> &phase_values, const std::size_t active_rank,
    const Real singular_tolerance = Real{}, const Real condition_tolerance = Real{}
)
{
    using namespace nmfd::operations::linalg;

    slice_projection_result<Real, MaxRank> result;
    result.active_rank = active_rank;
    result.alpha.resize( active_rank );
    result.alpha.fill( Real{} );

    if ( active_rank == 0 )
        return result;

    if ( slice_matrix.rows() != active_rank || slice_matrix.cols() != active_rank ||
         phase_values.size() != active_rank )
    {
        result.status            = small_solve_status::invalid_size;
        result.solve_info.status = result.status;
        result.solve_info.size   = active_rank;
        return result;
    }

    result.solve_info = solve( slice_matrix, phase_values, result.alpha, singular_tolerance, condition_tolerance );
    result.status     = result.solve_info.status;
    return result;
}

} // namespace symmetry

#endif
