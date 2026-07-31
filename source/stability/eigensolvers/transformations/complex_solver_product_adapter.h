#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_SOLVER_PRODUCT_ADAPTER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_SOLVER_PRODUCT_ADAPTER_H__

#include <cstddef>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<
    class ProductVectorSpace,
    class ComplexVectorSpace,
    class ComplexVectorBridge,
    class ComplexSolver>
class complex_solver_product_adapter
{
public:
    using product_space_type = ProductVectorSpace;
    using complex_space_type = ComplexVectorSpace;
    using bridge_type = ComplexVectorBridge;
    using solver_type = ComplexSolver;
    using vector_type = typename product_space_type::vector_type;

    complex_solver_product_adapter(
        const product_space_type& product_space,
        const complex_space_type& complex_space,
        const bridge_type& bridge,
        const solver_type& solver)
        : complex_space_(complex_space),
          bridge_(bridge),
          solver_(solver),
          complex_right_(complex_space, true, true),
          complex_solution_(complex_space, true, true)
    {
        (void)product_space;
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++solve_calls_;
        bridge_.pack(right_hand_side, *complex_right_);
        complex_space_.assign_scalar(
            typename complex_space_type::scalar_type{},
            *complex_solution_);
        if(
            !nmfd::solvers::krylov::solve_operator(
                solver_,
                *complex_right_,
                *complex_solution_))
        {
            ++failed_solves_;
            return false;
        }
        bridge_.unpack(*complex_solution_, solution);
        return true;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

    std::size_t failed_solves() const
    {
        return failed_solves_;
    }

private:
    const complex_space_type& complex_space_;
    const bridge_type& bridge_;
    const solver_type& solver_;
    mutable nmfd::detail::vector_wrap<
        complex_space_type,
        true,
        true> complex_right_;
    mutable nmfd::detail::vector_wrap<
        complex_space_type,
        true,
        true> complex_solution_;
    mutable std::size_t solve_calls_ = 0;
    mutable std::size_t failed_solves_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
