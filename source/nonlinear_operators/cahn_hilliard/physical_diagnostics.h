#ifndef NONLINEAR_OPERATORS_CH_PHYSICAL_DIAGNOSTICS_H
#define NONLINEAR_OPERATORS_CH_PHYSICAL_DIAGNOSTICS_H

#include "model.h"
#include <array>
#include <limits>
#include <nmfd/detail/vector_wrap.h>

namespace nonlinear_operators::cahn_hilliard
{
// Method-neutral endpoint diagnostics. D uses the physical chemical potential,
// not the convex-split stage potential. This observer never accepts/rejects steps.
template <class Space>
class physical_diagnostics
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using model_type  = model<scalar_type>;
    using wrap_type   = nmfd::detail::vector_wrap<Space>;
    struct report_type
    {
        scalar_type energy, mass, dissipation;
    };
    physical_diagnostics( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> physics )
        : ops_( detail::checked_space( std::move( ops ), physics ) ), model_( std::move( physics ) ),
          potential_( *ops_ ), work_( *ops_ )
    {
        potential_.start_use();
        work_.start_use();
    }
    struct mapping
    {
        model_type          model;
        vector_type         state, out;
        bool                energy;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const scalar_type c = state( i ), lap = model.grid.apply( state, i, 0, 1 );
            out( i ) = energy ? model.energy_density( c, lap ) : model.physical_potential( c, lap );
        }
    };
    struct laplacian_mapping
    {
        model_type          model;
        vector_type         in, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = model.grid.apply( in, i, 0, 1 );
        }
    };
    report_type evaluate( const vector_type &state )
    {
        if ( !ops_->is_valid_number( state ) )
            throw std::invalid_argument( "Nonfinite CH physical state" );
        typename Space::for_each_type each;
        const auto                    n    = model_->grid.size();
        const auto                    area = model_->cell_area();
        report_type                   result{};
        result.mass = area * ops_->sum( state );
        each( mapping{ *model_, state, *work_, true }, n );
        each.wait();
        result.energy = area * ops_->sum( *work_ );
        each( mapping{ *model_, state, *potential_, false }, n );
        each.wait();
        each( laplacian_mapping{ *model_, *potential_, *work_ }, n );
        each.wait();
        result.dissipation = -area * model_->mobility * ops_->scalar_prod( *potential_, *work_ );
        const std::array<scalar_type, 3> values{ result.energy, result.mass, result.dissipation };
        for ( auto value : values )
            if ( !std::isfinite( value ) )
                throw std::overflow_error( "Nonfinite CH physical diagnostics" );
        return result;
    }

private:
    std::shared_ptr<Space>            ops_;
    std::shared_ptr<const model_type> model_;
    wrap_type                         potential_, work_;
};
}
#endif
