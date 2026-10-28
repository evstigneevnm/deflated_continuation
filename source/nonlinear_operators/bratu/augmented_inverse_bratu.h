#ifndef __AUGMENTED_INVERSE_BRATU_H__
#define __AUGMENTED_INVERSE_BRATU_H__

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <nmfd/operations/dense_operations_base.h>
#include <scfd/backend/serial_cpu.h>

namespace nonlinear_operators
{

// Assemble the already formulated Jacobian. Pivot the complete system, not its state block.
template <class Space, class Operator>
class augmented_inverse_bratu
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;

    explicit augmented_inverse_bratu( std::shared_ptr<Space> space )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing Bratu pair space" ) ),
          size_( space_->first()->get_vector_size() ), matrix_( ( size_ + 1 ) * ( size_ + 1 ) ),
          work_( matrix_.size() ), rhs_( size_ + 1 )
    {
    }

    void set_operator( std::shared_ptr<const Operator> op )
    {
        if ( !op )
            throw std::invalid_argument( "Missing Bratu bordered Jacobian" );
        ready_ = false;
        op->state_operator->copy_matrix( matrix_.data() );
        const auto n = size_ + 1;
        // Expand the compact state matrix backwards, without another matrix-sized buffer.
        for ( std::size_t i = size_; i-- > 0; )
            for ( std::size_t j = size_; j-- > 0; )
                matrix_[i * n + j] = matrix_[i * size_ + j];
        space_->first()->get( *op->parameter_column, rhs_.data(), size_ );
        for ( std::size_t i = 0; i < size_; ++i )
            matrix_[i * n + size_] = rhs_[i];
        space_->first()->get( *op->constraint_row, rhs_.data(), size_ );
        for ( std::size_t j = 0; j < size_; ++j )
            matrix_[size_ * n + j] = rhs_[j];
        matrix_.back() = op->constraint_parameter;
        ready_         = true;
    }

    bool apply( vector_type &in_out ) const
    {
        if ( !ready_ )
            return false;
        std::copy( matrix_.begin(), matrix_.end(), work_.begin() );
        space_->first()->get( in_out.first, rhs_.data(), size_ );
        rhs_.back()      = in_out.second[0];
        using dense_type = nmfd::operations::dense_operations<scalar_type, scfd::backend::serial_cpu<std::ptrdiff_t>>;
        scfd::arrays::array_nd<scalar_type, 2, scfd::memory::host, scfd::arrays::last_index_fast_arranger> matrix;
        scfd::arrays::array<scalar_type, scfd::memory::host>                                               rhs;
        matrix.init_by_raw_data( work_.data(), size_ + 1, size_ + 1 );
        rhs.init_by_raw_data( rhs_.data(), size_ + 1 );
        if ( !dense_type::solve_in_place( matrix, rhs, size_ + 1 ) )
            return false;
        space_->first()->set( rhs_.data(), in_out.first, size_ );
        in_out.second[0] = rhs_.back();
        return true;
    }

private:
    std::shared_ptr<Space>           space_;
    std::size_t                      size_;
    std::vector<scalar_type>         matrix_;
    mutable std::vector<scalar_type> work_, rhs_;
    bool                             ready_ = false;
};

} // namespace nonlinear_operators

#endif
