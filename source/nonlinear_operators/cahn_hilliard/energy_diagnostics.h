#ifndef NONLINEAR_OPERATORS_CH_ENERGY_DIAGNOSTICS_H
#define NONLINEAR_OPERATORS_CH_ENERGY_DIAGNOSTICS_H

#include "model.h"
#include <array>
#include <limits>
#include <nmfd/detail/vector_wrap.h>

namespace nonlinear_operators::cahn_hilliard
{
// Optional observer for the forward convex-split Euler update, not a controller
// and not an assertion of energy stability for arbitrary IMEX tables. Scratch
// is backend-resident; evaluations return scalar diagnostics only.
template <class Space>
class energy_diagnostics
{
public:
    using scalar_type = typename Space::scalar_type;
    using norm_type   = typename Space::norm_type;
    using vector_type = typename Space::vector_type;
    using model_type  = model<scalar_type>;
    using wrap_type   = nmfd::detail::vector_wrap<Space>;
    struct report_type
    {
        scalar_type energy_before, energy_after, mass_before, mass_after;
        scalar_type dissipation, residual_work, dissipation_bound_defect;
        norm_type   step_defect_norm, roundoff_budget;
    };
    energy_diagnostics( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> physics )
        : ops_( detail::checked_space( std::move( ops ), physics ) ), model_( std::move( physics ) ),
          potential_( *ops_ ), laplacian_( *ops_ ), work_( *ops_ )
    {
        potential_.start_use();
        laplacian_.start_use();
        work_.start_use();
    }
    struct energy_mapping
    {
        model_type          model;
        vector_type         state, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = model.energy_density( state( i ), model.grid.apply( state, i, 0, 1 ) );
        }
    };
    struct potential_mapping
    {
        model_type          model;
        vector_type         before, after, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = model.convex_potential( after( i ), model.grid.apply( after, i, 0, 1 ) ) - before( i );
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
    struct defect_mapping
    {
        vector_type         before, after, laplacian, out;
        scalar_type         h, mobility;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = ( after( i ) - before( i ) ) - h * mobility * laplacian( i );
        }
    };
    report_type evaluate_step( const vector_type &before, const vector_type &after, scalar_type h )
    {
        if ( !std::isfinite( h ) || h <= 0 )
            throw std::invalid_argument( "CH energy diagnostics require a positive finite step" );
        if ( !ops_->is_valid_number( before ) || !ops_->is_valid_number( after ) )
            throw std::invalid_argument( "Nonfinite CH energy state" );
        typename Space::for_each_type each;
        const auto                    n    = model_->grid.size();
        const scalar_type             area = model_->cell_area();
        report_type                   result{};
        result.mass_before = area * ops_->sum( before );
        result.mass_after  = area * ops_->sum( after );
        each( energy_mapping{ *model_, before, *work_ }, n );
        each.wait();
        result.energy_before = area * ops_->sum( *work_ );
        each( energy_mapping{ *model_, after, *work_ }, n );
        each.wait();
        result.energy_after = area * ops_->sum( *work_ );
        each( potential_mapping{ *model_, before, after, *potential_ }, n );
        each.wait();
        each( laplacian_mapping{ *model_, *potential_, *laplacian_ }, n );
        each.wait();
        result.dissipation = -area * model_->mobility * ops_->scalar_prod( *potential_, *laplacian_ );
        each( defect_mapping{ before, after, *laplacian_, *work_, h, model_->mobility }, n );
        each.wait();
        result.step_defect_norm         = ops_->norm( *work_ ) / std::sqrt( scalar_type( n ) );
        result.residual_work            = area * ops_->scalar_prod( *potential_, *work_ );
        result.dissipation_bound_defect = result.energy_after - result.energy_before + h * result.dissipation;
        // Roundoff allowance, not a nonlinear-solver acceptance tolerance. The
        // tests independently bound the step defect and residual work.
        result.roundoff_budget = scalar_type( 256 ) * std::numeric_limits<norm_type>::epsilon() *
                                 ( 1 + std::abs( result.energy_before ) + std::abs( result.energy_after ) +
                                   std::abs( h * result.dissipation ) + std::abs( result.residual_work ) );
        const std::array<scalar_type, 9> values{
            result.energy_before,
            result.energy_after,
            result.mass_before,
            result.mass_after,
            result.dissipation,
            result.residual_work,
            result.dissipation_bound_defect,
            scalar_type( result.step_defect_norm ),
            scalar_type( result.roundoff_budget )
        };
        for ( const auto value : values )
            if ( !std::isfinite( value ) )
                throw std::overflow_error( "Nonfinite CH energy diagnostics" );
        return result;
    }

private:
    std::shared_ptr<Space>            ops_;
    std::shared_ptr<const model_type> model_;
    wrap_type                         potential_, laplacian_, work_;
};
}
#endif
