#ifndef PDE_VALIDATION_ODE_PROBLEM_H
#define PDE_VALIDATION_ODE_PROBLEM_H

#include "diffusion.h"
#include <memory>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/runge_kutta/imex_operator.h>

namespace nonlinear_operators::pde_validation
{
// A pure rate provider owns no stage buffers or Jacobian storage. Explicit
// integration/IMEX explicit partitions do not allocate implicit workspace.
template <class Space, class Model, partition Part = partition::full>
class rate_problem
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    rate_problem( std::shared_ptr<Space> ops, std::shared_ptr<const Model> model )
        : ops_( std::move( ops ) ), model_( std::move( model ) )
    {
        if ( !ops_ || !model_ || ops_->size() != Model::components * model_->grid.size() )
            throw std::invalid_argument( "PDE rate space/model mismatch" );
    }
    void set_time( scalar_type time )
    {
        time_ = time;
    }
    struct mapping
    {
        Model               model;
        vector_type         state, out;
        scalar_type         time;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            scalar_type values[Model::components];
            model.template rate<Part>( state, i, time, values );
            for ( int c = 0; c < Model::components; ++c )
                out( Model::components * i + c ) = values[c];
        }
    };
    bool evaluate_rate( const vector_type &state, vector_type &out ) const
    {
        typename Space::for_each_type each;
        each( mapping{ *model_, state, out, time_ }, model_->grid.size() );
        each.wait();
        return true;
    }

private:
    std::shared_ptr<Space>       ops_;
    std::shared_ptr<const Model> model_;
    scalar_type                  time_ = 0;
};

// Identity-mass facade. Model/geometry are shared immutable data; each facade
// owns its time, shifted-stage buffers and frozen Jacobian independently.
template <class Space, class Model, partition Part = partition::full>
class ode_problem
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using wrap        = nmfd::detail::vector_wrap<Space>;
    struct jacobi_operator_type
    {
        std::shared_ptr<Space>       ops;
        std::shared_ptr<const Model> model;
        wrap                         point;
        scalar_type                  a = 0, b = 1, time = 0;
        jacobi_operator_type( std::shared_ptr<Space> o, std::shared_ptr<const Model> m )
            : ops( std::move( o ) ), model( std::move( m ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            Model               model;
            vector_type         point, in, out;
            scalar_type         a, b, time;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                scalar_type value[Model::components];
                model.template jacobian<Part>( point, in, i, time, value );
                for ( int c = 0; c < Model::components; ++c )
                    out( Model::components * i + c ) = a * in( Model::components * i + c ) - b * value[c];
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Space::for_each_type each;
            each( mapping{ *model, *point, in, out, a, b, time }, model->grid.size() );
            each.wait();
        }
    };
    ode_problem( std::shared_ptr<Space> ops, std::shared_ptr<const Model> model )
        : ops_( ops ? std::move( ops ) : throw std::invalid_argument( "Missing PDE space" ) ),
          model_( std::move( model ) ), previous_( *ops_ ), rhs_( *ops_ ),
          jacobian_( std::make_shared<jacobi_operator_type>( ops_, model_ ) )
    {
        if ( !model_ || ops_->size() != Model::components * model_->grid.size() )
            throw std::invalid_argument( "PDE space/model mismatch" );
        previous_.start_use();
        rhs_.start_use();
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
    void set_time( scalar_type t )
    {
        time_ = t;
    }
    void set_affine_shift( scalar_type a, scalar_type b )
    {
        if ( !std::isfinite( a ) || !std::isfinite( b ) || b <= 0 )
            throw std::invalid_argument( "Invalid PDE affine shift" );
        a_ = a;
        b_ = b;
    }
    enum class evaluation
    {
        residual,
        rate,
        partial_time
    };
    template <evaluation Mode>
    struct mapping
    {
        Model               model;
        vector_type         state, previous, rhs, out;
        scalar_type         time, a, b;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            scalar_type value[Model::components];
            if constexpr ( Mode == evaluation::partial_time )
                model.template time_derivative<Part>( i, time, value );
            else
                model.template rate<Part>( state, i, time, value );
            for ( int c = 0; c < Model::components; ++c )
            {
                const auto j = Model::components * i + c;
                if constexpr ( Mode == evaluation::residual )
                    out( j ) = a * ( state( j ) - previous( j ) ) - b * ( value[c] + rhs( j ) );
                else
                    out( j ) = value[c];
            }
        }
    };
    void apply( const vector_type &state, vector_type &out ) const
    {
        evaluate<evaluation::residual>( state, out );
    }
    bool evaluate_rate( const vector_type &state, vector_type &out ) const
    {
        evaluate<evaluation::rate>( state, out );
        return true;
    }
    bool evaluate_time_derivative( const vector_type &state, vector_type &out ) const
    {
        evaluate<evaluation::partial_time>( state, out );
        return true;
    }
    void set_linearization_point( const vector_type &state )
    {
        ops_->assign( state, *jacobian_->point );
        jacobian_->time = time_;
        jacobian_->a    = a_;
        jacobian_->b    = b_;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian_;
    }

private:
    template <evaluation Mode>
    void evaluate( const vector_type &state, vector_type &out ) const
    {
        typename Space::for_each_type each;
        each( mapping<Mode>{ *model_, state, *previous_, *rhs_, out, time_, a_, b_ }, model_->grid.size() );
        each.wait();
    }
    std::shared_ptr<Space>                ops_;
    std::shared_ptr<const Model>          model_;
    wrap                                  previous_, rhs_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    scalar_type                           time_ = 0, a_ = 0, b_ = 1;
};

template <class Space, class Model>
class imex_problem
{
public:
    using explicit_type      = rate_problem<Space, Model, partition::explicit_part>;
    using implicit_type      = ode_problem<Space, Model, partition::implicit_part>;
    using imex_operator_type = nmfd::time_steppers::runge_kutta::imex_operator<explicit_type, implicit_type>;
    imex_problem( std::shared_ptr<Space> ops, std::shared_ptr<const Model> model )
        : bundle_(
              std::make_shared<imex_operator_type>( imex_operator_type{
                  std::make_shared<explicit_type>( ops, model ), std::make_shared<implicit_type>( ops, model )
              } )
          )
    {
    }
    std::shared_ptr<imex_operator_type> get_imex_operator() const
    {
        return bundle_;
    }

private:
    std::shared_ptr<imex_operator_type> bundle_;
};
}
#endif
