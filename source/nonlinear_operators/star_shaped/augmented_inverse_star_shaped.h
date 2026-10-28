#ifndef __AUGMENTED_INVERSE_STAR_SHAPED_H__
#define __AUGMENTED_INVERSE_STAR_SHAPED_H__

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/dense_operations_base.h>
#include <scfd/backend/serial_cpu.h>

namespace nonlinear_operators
{

// Pivot the complete scalar-state bordered Jacobian, including at folds.
template <class Space, class Operator>
class augmented_inverse_star_shaped
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;

    explicit augmented_inverse_star_shaped( std::shared_ptr<Space> space )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing star-shaped pair space" ) ),
          basis_( *space_->first() ), work_( *space_->first() )
    {
        if ( space_->first()->get_vector_size() != 1 )
            throw std::invalid_argument( "Star-shaped requires one state unknown" );
        basis_.init();
        basis_.start_use();
        work_.init();
        work_.start_use();
        space_->first()->assign_scalar( scalar_type( 1 ), *basis_ );
    }

    void set_operator( std::shared_ptr<const Operator> op )
    {
        ready_ = false;
        if ( !op )
            throw std::invalid_argument( "Missing star-shaped bordered Jacobian" );
        op->state_operator->apply( *basis_, *work_ );
        space_->first()->get( *work_, &matrix_[0], 1 );
        space_->first()->get( *op->parameter_column, &matrix_[1], 1 );
        space_->first()->get( *op->constraint_row, &matrix_[2], 1 );
        matrix_[3] = op->constraint_parameter;
        ready_     = true;
    }

    bool apply( vector_type &in_out ) const
    {
        if ( !ready_ )
            return false;
        auto                       matrix_values = matrix_;
        std::array<scalar_type, 2> rhs_values;
        space_->first()->get( in_out.first, rhs_values.data(), 1 );
        rhs_values[1]    = in_out.second[0];
        using dense_type = nmfd::operations::dense_operations<scalar_type, scfd::backend::serial_cpu<std::ptrdiff_t>>;
        scfd::arrays::array_nd<scalar_type, 2, scfd::memory::host, scfd::arrays::last_index_fast_arranger> matrix;
        scfd::arrays::array<scalar_type, scfd::memory::host>                                               rhs;
        matrix.init_by_raw_data( matrix_values.data(), 2, 2 );
        rhs.init_by_raw_data( rhs_values.data(), 2 );
        if ( !dense_type::solve_in_place( matrix, rhs, 2 ) )
            return false;
        space_->first()->set( rhs_values.data(), in_out.first, 1 );
        in_out.second[0] = rhs_values[1];
        return true;
    }

private:
    std::shared_ptr<Space>                                                    space_;
    nmfd::detail::vector_wrap<typename Space::first_space_type, false, false> basis_, work_;
    std::array<scalar_type, 4>                                                matrix_{};
    bool                                                                      ready_ = false;
};

}

#endif
