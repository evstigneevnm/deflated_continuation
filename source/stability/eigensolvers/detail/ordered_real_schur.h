#ifndef __STABILITY_EIGENSOLVERS_DETAIL_ORDERED_REAL_SCHUR_H__
#define __STABILITY_EIGENSOLVERS_DETAIL_ORDERED_REAL_SCHUR_H__

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../eigenvalue_target.h"

namespace stability
{
namespace eigensolvers
{
namespace detail
{

template<class Real, class SchurBlock>
std::complex<Real> representative_eigenvalue(
    const SchurBlock& block,
    const eigenvalue_target<Real>& target)
{
    if(block.eigenvalues.empty())
        throw std::logic_error("empty real Schur block");

    std::complex<Real> result = block.eigenvalues.front();
    for(std::size_t index = 1; index < block.eigenvalues.size(); ++index)
    {
        if(target.precedes(block.eigenvalues[index], result))
            result = block.eigenvalues[index];
    }
    return result;
}

template<class Real, class Lapack, class SchurDecomposition>
std::size_t order_leading_real_schur_blocks(
    const Lapack& lapack,
    SchurDecomposition& decomposition,
    const eigenvalue_target<Real>& target,
    std::size_t requested_dimension)
{
    const std::size_t dimension = decomposition.quasi_triangular.rows();
    if(requested_dimension == 0 || requested_dimension > dimension)
        throw std::invalid_argument("invalid retained Schur dimension");

    std::size_t retained = 0;
    while(retained < requested_dimension)
    {
        const auto blocks = lapack.real_schur_blocks(decomposition);
        std::size_t selected = blocks.size();
        std::complex<Real> selected_value{};

        for(std::size_t index = 0; index < blocks.size(); ++index)
        {
            if(blocks[index].first < retained)
                continue;
            const auto value =
                representative_eigenvalue<Real>(blocks[index], target);
            if(
                selected == blocks.size() ||
                target.precedes(value, selected_value))
            {
                selected = index;
                selected_value = value;
            }
        }
        if(selected == blocks.size())
            throw std::logic_error("unable to select a real Schur block");

        if(blocks[selected].first != retained)
        {
            lapack.move_schur_block(
                decomposition,
                blocks[selected].first,
                retained);
        }

        const auto reordered_blocks = lapack.real_schur_blocks(decomposition);
        std::size_t leading = reordered_blocks.size();
        for(std::size_t index = 0; index < reordered_blocks.size(); ++index)
        {
            if(reordered_blocks[index].first == retained)
            {
                leading = index;
                break;
            }
        }
        if(leading == reordered_blocks.size())
            throw std::logic_error("moved Schur block was not found");
        retained += reordered_blocks[leading].size;
    }
    return retained;
}

} // namespace detail
} // namespace eigensolvers
} // namespace stability

#endif
