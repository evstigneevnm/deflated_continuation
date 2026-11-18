#ifndef NONLINEAR_OPERATORS_CH_DETAIL_CONCENTRATION_PROBLEM_H
#define NONLINEAR_OPERATORS_CH_DETAIL_CONCENTRATION_PROBLEM_H

#include "../model.h"
#include <nmfd/detail/vector_wrap.h>

namespace nonlinear_operators::cahn_hilliard::detail
{
// Shared identity-mass stage machinery. ConvexOnly selects the implicit IMEX
// partition; otherwise the rate includes the complete physical potential.
// Both use two backend passes and independent residual/prepared-Jacobian scratch.
template <class Space, bool ConvexOnly>
class concentration_problem
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using model_type  = model<scalar_type>;
    using wrap_type   = nmfd::detail::vector_wrap<Space>;
    struct jacobi_operator_type
    {
        std::shared_ptr<Space>            ops;
        std::shared_ptr<const model_type> physics;
        wrap_type                         derivative;
        mutable wrap_type                 work;
        scalar_type                       a = 0, b = 1;
        jacobi_operator_type( std::shared_ptr<Space> o, std::shared_ptr<const model_type> m )
            : ops( std::move( o ) ), physics( std::move( m ) ), derivative( *ops ), work( *ops )
        {
            derivative.start_use();
            work.start_use();
        }
        struct prepare_mapping
        {
            vector_type         state, out;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( i ) = 3 * state( i ) * state( i );
                if constexpr ( !ConvexOnly )
                    out( i ) -= 1;
            }
        };
        struct potential_mapping
        {
            model_type          model;
            vector_type         derivative, in, out;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( i ) = derivative( i ) * in(i)-model.epsilon_squared * model.grid.apply( in, i, 0, 1 );
            }
        };
        struct action_mapping
        {
            model_type          model;
            vector_type         in, work, out;
            scalar_type         a, b;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( i ) = a * in(i)-b * model.mobility * model.grid.apply( work, i, 0, 1 );
            }
        };
        void prepare( const vector_type &state, scalar_type mass, scalar_type rhs )
        {
            typename Space::for_each_type each;
            each( prepare_mapping{ state, *derivative }, physics->grid.size() );
            each.wait();
            a = mass;
            b = rhs;
        }
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Space::for_each_type each;
            each( potential_mapping{ *physics, *derivative, in, *work }, physics->grid.size() );
            each.wait();
            each( action_mapping{ *physics, in, *work, out, a, b }, physics->grid.size() );
            each.wait();
        }
    };
    concentration_problem( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> physics )
        : ops_( checked_space( std::move( ops ), physics ) ), model_( std::move( physics ) ), previous_( *ops_ ),
          rhs_( *ops_ ), potential_( *ops_ ), jacobian_( std::make_shared<jacobi_operator_type>( ops_, model_ ) )
    {
        previous_.start_use();
        rhs_.start_use();
        potential_.start_use();
    }
    vector_type &previous_state()
    {
        return *previous_;
    }
    const vector_type &previous_state() const
    {
        return *previous_;
    }
    vector_type &known_rhs()
    {
        return *rhs_;
    }
    const vector_type &known_rhs() const
    {
        return *rhs_;
    }
    void set_time( scalar_type )
    {
    } // Autonomous; rates are independent of stage context.
    void set_affine_shift( scalar_type a, scalar_type b )
    {
        if ( !std::isfinite( a ) || !std::isfinite( b ) || b <= 0 )
            throw std::invalid_argument( "Invalid CH affine shift" );
        a_ = a;
        b_ = b;
    }
    struct potential_mapping
    {
        model_type          model;
        vector_type         state, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            if constexpr ( ConvexOnly )
                out( i ) = model.convex_potential( state( i ), model.grid.apply( state, i, 0, 1 ) );
            else
                out( i ) = model.physical_potential( state( i ), model.grid.apply( state, i, 0, 1 ) );
        }
    };
    template <bool Rate>
    struct residual_mapping
    {
        model_type          model;
        vector_type         state, previous, rhs, potential, out;
        scalar_type         a, b;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const scalar_type rate = model.mobility * model.grid.apply( potential, i, 0, 1 );
            if constexpr ( Rate )
                out( i ) = rate;
            else
                out( i ) = a * ( state( i ) - previous( i ) ) - b * ( rate + rhs( i ) );
        }
    };
    void apply( const vector_type &state, vector_type &residual ) const
    {
        evaluate<false>( state, residual );
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        evaluate<true>( state, rate );
        return true;
    }
    void set_linearization_point( const vector_type &state )
    {
        jacobian_->prepare( state, a_, b_ );
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian_;
    }

private:
    template <bool Rate>
    void evaluate( const vector_type &state, vector_type &out ) const
    {
        typename Space::for_each_type each;
        each( potential_mapping{ *model_, state, *potential_ }, model_->grid.size() );
        each.wait();
        each(
            residual_mapping<Rate>{ *model_, state, *previous_, *rhs_, *potential_, out, a_, b_ }, model_->grid.size()
        );
        each.wait();
    }
    std::shared_ptr<Space>                ops_;
    std::shared_ptr<const model_type>     model_;
    wrap_type                             previous_, rhs_;
    mutable wrap_type                     potential_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    scalar_type                           a_ = 0, b_ = 1;
};
}
#endif
