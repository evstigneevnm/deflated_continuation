#ifndef PERIODIC_ORBIT_PREPARED_BDF_SECTION_H
#define PERIODIC_ORBIT_PREPARED_BDF_SECTION_H

#include <array>
#include <nmfd/time_steppers/bdf/continuous_integration.h>
#include <periodic_orbit/flows/implicit_stage_provider.h>
#include <periodic_orbit/flows/prepared_section.h>

namespace periodic_orbit
{
struct bdf_flight_method
{
    unsigned int maximum_order = 2;
};
template <class T>
struct bdf_step_record
{
    T            time{}, dt{}, end{}, theta = T( 1 ), candidate_end{};
    unsigned int order = 1;
    template <class Stepper>
    static bdf_step_record capture( const Stepper &step, T time, T dt, T end, T theta, T candidate_end )
    {
        return { time, dt, end, theta, candidate_end, step.get_order() };
    }
};

// Replay only the fixed BDF1/BDF2 recurrence, with cold history per flight.
// The order and signed mesh are data; no controller decisions are differentiated.
template <class Space, class StageProvider>
class bdf_flight_replay
{
public:
    using T           = typename Space::scalar_type;
    using X           = typename Space::vector_type;
    using record_type = bdf_step_record<T>;
    using wrap        = nmfd::detail::vector_wrap<Space, false, false>;
    bdf_flight_replay(
        std::shared_ptr<Space> space, std::shared_ptr<StageProvider> provider, const bdf_flight_method &method = {},
        std::size_t maximum_steps = 1000000
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing BDF replay space" ) ),
          provider_( std::move( provider ) ), method_( method ), maximum_steps_( maximum_steps ), state_( *space_ ),
          tangent_( *space_ ), older_( *space_ ), delta_older_( *space_ ), stage_( *space_ ), delta_stage_( *space_ ),
          reference_( *space_ ), delta_reference_( *space_ ), endpoint_( *space_ ), delta_endpoint_( *space_ ),
          time_rate_( *space_ )
    {
        if ( !provider_ || !supports( method ) || !maximum_steps )
            throw std::invalid_argument( "Invalid fixed BDF replay configuration" );
        for ( auto *x :
              { &state_, &tangent_, &stage_, &delta_stage_, &reference_, &delta_reference_, &endpoint_,
                &delta_endpoint_, &time_rate_ } )
        {
            x->init();
            x->start_use();
        }
        if ( method.maximum_order == 2 )
            for ( auto *x : { &older_, &delta_older_ } )
            {
                x->init();
                x->start_use();
            }
    }
    static bool supports( const bdf_flight_method &method )
    {
        return method.maximum_order == 1 || method.maximum_order == 2;
    }
    static void
    validate( const bdf_flight_method &method, const std::vector<record_type> &steps, std::size_t maximum_steps )
    {
        if ( !supports( method ) )
            throw std::invalid_argument( "Only fixed BDF1/BDF2 replay is supported" );
        detail::validate_flight_schedule( steps, maximum_steps );
        for ( std::size_t i = 0; i < steps.size(); ++i )
        {
            const auto &r         = steps[i];
            const T     tolerance = T( 4 ) * std::numeric_limits<T>::epsilon() *
                                std::max( { std::abs( r.time ), std::abs( r.candidate_end ), std::abs( r.dt ) } );
            const bool paired =
                i && ( equal_step( r.dt, steps[i - 1].dt ) || std::abs( r.dt - steps[i - 1].dt ) <= tolerance );
            // Honor a recorded restart. In particular, the step following a
            // roundoff-clipped step can restart even though both h are close.
            const bool valid_order = r.order == 1 || ( r.order == 2 && method.maximum_order == 2 && paired );
            if ( !valid_order || !std::isfinite( r.candidate_end ) ||
                 std::abs( r.candidate_end - ( r.time + r.dt ) ) > tolerance ||
                 ( r.theta == T( 1 ) && r.end != r.candidate_end ) ||
                 ( r.dt > 0 ? r.end > r.candidate_end : r.end < r.candidate_end ) )
                throw std::invalid_argument( "Invalid fixed BDF order/history record" );
        }
    }
    void
    run( const bdf_flight_method &method, const std::vector<record_type> &steps, const X &initial, const X *direction,
         X &endpoint, X *derivative, X *time_rate )
    {
        validate( method, steps, maximum_steps_ );
        if ( method.maximum_order != method_.maximum_order || bool( direction ) != bool( derivative ) ||
             &endpoint == derivative || &endpoint == time_rate || ( derivative && derivative == time_rate ) )
            throw std::invalid_argument( "Incompatible BDF replay outputs or method" );
        for ( const auto *x : std::array<const X *, 5>{ &initial, &endpoint, direction, derivative, time_rate } )
            if ( x && space_->get_size( *x ) != space_->get_vector_size() )
                throw std::invalid_argument( "BDF replay vector shape mismatch" );
        if ( !space_->is_valid_number( initial ) || ( direction && !space_->is_valid_number( *direction ) ) )
            throw std::invalid_argument( "Nonfinite BDF replay input" );
        space_->assign( initial, *state_ );
        if ( direction )
            space_->assign( *direction, *tangent_ );
        for ( std::size_t i = 0; i < steps.size(); ++i )
        {
            const auto &r = steps[i];
            if ( r.order == 1 )
            {
                space_->assign( *state_, *reference_ );
                if ( direction )
                    space_->assign( *tangent_, *delta_reference_ );
            }
            else
            {
                space_->assign_lin_comb( T( 4 ) / 3, *state_, T( -1 ) / 3, *older_, *reference_ );
                if ( direction )
                    space_->assign_lin_comb( T( 4 ) / 3, *tangent_, T( -1 ) / 3, *delta_older_, *delta_reference_ );
            }
            // Native BDF starts Newton from the committed input, not from w.
            space_->assign( *state_, *stage_ );
            const T alpha = ( r.order == 2 ? T( 1.5 ) : T( 1 ) ) / r.dt;
            if ( !std::isfinite( alpha ) || !provider_->solve_stage( r.candidate_end, alpha, *reference_, *stage_ ) ||
                 !space_->is_valid_number( *stage_ ) )
                throw std::runtime_error( "BDF base replay solve failed" );
            if ( direction && ( !provider_->solve_tangent( *stage_, *reference_, *delta_reference_, *delta_stage_ ) ||
                                !space_->is_valid_number( *delta_stage_ ) ) )
                throw std::runtime_error( "BDF history tangent solve failed" );
            space_->assign( *stage_, *endpoint_ );
            if ( direction )
                space_->assign( *delta_stage_, *delta_endpoint_ );
            const bool last = i + 1 == steps.size();
            if ( last )
            {
                const T previous_h = i ? steps[i - 1].dt : T( 0 );
                if ( r.theta != T( 1 ) )
                {
                    const auto c =
                        nmfd::time_steppers::bdf::make_interpolation( r.order, r.theta, r.dt, { previous_h } );
                    interpolate( c, *state_, *stage_, older_, *endpoint_, r.order );
                    if ( direction )
                        interpolate( c, *tangent_, *delta_stage_, delta_older_, *delta_endpoint_, r.order );
                }
                if ( time_rate )
                {
                    const T ratio           = r.order == 2 ? r.dt / previous_h : T( 0 );
                    const T endpoint_weight = r.order == 1 ? T( 1 ) : ( T( 2 ) * r.theta * ratio + 1 ) / ( ratio + 1 );
                    const T history_weight  = r.order == 1 ? T( 0 ) : ratio * ( T( 1 ) - endpoint_weight );
                    space_->assign_lin_comb( T( 1 ), *stage_, T( -1 ), *state_, *time_rate_ );
                    space_->scale( endpoint_weight / r.dt, *time_rate_ );
                    if ( r.order == 2 )
                    {
                        space_->assign_lin_comb( T( 1 ), *state_, T( -1 ), *older_, *reference_ );
                        space_->add_lin_comb( history_weight / r.dt, *reference_, *time_rate_ );
                    }
                    if ( !space_->is_valid_number( *time_rate_ ) )
                        throw std::runtime_error( "Nonfinite BDF dense time derivative" );
                }
            }
            if ( !space_->is_valid_number( *endpoint_ ) ||
                 ( direction && !space_->is_valid_number( *delta_endpoint_ ) ) )
                throw std::runtime_error( "Nonfinite BDF replay endpoint" );
            if ( method.maximum_order == 2 )
            {
                space_->assign( *state_, *older_ );
                if ( direction )
                    space_->assign( *tangent_, *delta_older_ );
            }
            space_->assign( *endpoint_, *state_ );
            if ( direction )
                space_->assign( *delta_endpoint_, *tangent_ );
        }
        space_->assign( *state_, endpoint );
        if ( derivative )
            space_->assign( *tangent_, *derivative );
        if ( time_rate )
            space_->assign( *time_rate_, *time_rate );
    }

private:
    std::shared_ptr<Space>         space_;
    std::shared_ptr<StageProvider> provider_;
    bdf_flight_method              method_;
    std::size_t                    maximum_steps_;
    wrap state_, tangent_, older_, delta_older_, stage_, delta_stage_, reference_, delta_reference_, endpoint_,
        delta_endpoint_, time_rate_;
    static bool equal_step( T a, T b )
    {
        return std::abs( a - b ) <=
               T( 8 ) * std::numeric_limits<T>::epsilon() * std::max( std::abs( a ), std::abs( b ) );
    }
    void interpolate(
        const nmfd::time_steppers::bdf::interpolation_coefficients<T> &c, const X &state, const X &candidate,
        const wrap &older, X &out, unsigned int order
    ) const
    {
        space_->assign_lin_comb( T( 1 ) - c.endpoint + c.differences[0], state, c.endpoint, candidate, out );
        if ( order == 2 )
            space_->add_lin_comb( -c.differences[0], *older, out );
    }
};

template <class Space, class Provider, class Section>
using prepared_bdf_section = prepared_section<
    Space, bdf_flight_replay<Space, Provider>, Section, bdf_flight_method,
    bdf_step_record<typename Space::scalar_type>>;

template <class Space, class ProviderFactory, class Section>
class bdf_flight_factory
{
public:
    using provider_type = typename ProviderFactory::provider_type;
    using flow_type     = prepared_bdf_section<Space, provider_type, Section>;
    struct params
    {
        unsigned int maximum_order        = 2;
        std::size_t  maximum_replay_steps = 1000000;
    };
    bdf_flight_factory(
        std::shared_ptr<Space> space, std::shared_ptr<const ProviderFactory> providers, const params &p = {}
    )
        : space_( std::move( space ) ), providers_( std::move( providers ) ), params_( p )
    {
        if ( !space_ || !providers_ || !flow_type::replay_type::supports( { p.maximum_order } ) ||
             !p.maximum_replay_steps )
            throw std::invalid_argument( "Invalid fixed BDF flight factory" );
    }
    template <class Segments>
    auto make_workspace( const Segments &segments ) const
    {
        for ( const auto &s : segments )
        {
            if ( !s.flight )
                throw std::invalid_argument( "Missing BDF segment map" );
            flow_type::replay_type::validate(
                { params_.maximum_order }, s.flight->records(), params_.maximum_replay_steps
            );
        }
        return std::make_shared<typename flow_type::replay_type>(
            space_, providers_->make_provider(), bdf_flight_method{ params_.maximum_order },
            params_.maximum_replay_steps
        );
    }
    template <class Map>
    flow_type prepare(
        std::shared_ptr<typename flow_type::replay_type> workspace, const Map &flight,
        const typename Space::vector_type &initial, const typename Space::vector_type &arrival,
        std::shared_ptr<const Section> target, typename Space::scalar_type speed, typename Space::scalar_type tolerance
    ) const
    {
        flow_type::replay_type::validate( { params_.maximum_order }, flight.records(), params_.maximum_replay_steps );
        return {
            space_,
            std::move( workspace ),
            { params_.maximum_order },
            flight.records(),
            initial,
            arrival,
            std::move( target ),
            speed,
            tolerance
        };
    }

private:
    std::shared_ptr<Space>                 space_;
    std::shared_ptr<const ProviderFactory> providers_;
    params                                 params_;
};
}
#endif
