#ifndef PERIODIC_ORBIT_PREPARED_BDF_SECTION_H
#define PERIODIC_ORBIT_PREPARED_BDF_SECTION_H

#include <array>
#include <nmfd/time_steppers/bdf/history.h>
#include <nmfd/time_steppers/bdf/continuous_integration.h>
#include <periodic_orbit/flows/implicit_stage_provider.h>
#include <periodic_orbit/flows/prepared_section.h>

namespace periodic_orbit
{
struct bdf_flight_method
{
    unsigned int maximum_order      = 2;
    bool         adaptive           = false;
    double       minimum_step_ratio = .2, maximum_step_ratio = 2;
};
template <class T>
struct bdf_step_record
{
    T            time{}, dt{}, end{}, theta = T( 1 ), candidate_end{};
    unsigned int order    = 1;
    unsigned int attempts = 1;
    template <class Stepper>
    static bdf_step_record capture( const Stepper &step, T time, T dt, T end, T theta, T candidate_end )
    {
        return { time, dt, end, theta, candidate_end, step.get_order(), step.get_attempts() };
    }
};

// Replay the accepted BDF recurrence, with cold history per flight.
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
            throw std::invalid_argument( "Invalid BDF replay configuration" );
        for ( auto *x :
              { &state_, &tangent_, &stage_, &delta_stage_, &reference_, &delta_reference_, &endpoint_,
                &delta_endpoint_, &time_rate_ } )
        {
            x->init();
            x->start_use();
        }
        if ( method.maximum_order == 2 && !method.adaptive )
            for ( auto *x : { &older_, &delta_older_ } )
            {
                x->init();
                x->start_use();
            }
        if ( method.adaptive )
        {
            history_       = std::make_unique<history_type>( *space_ );
            delta_history_ = std::make_unique<history_type>( *space_ );
            history_->initialize( method.maximum_order );
            delta_history_->initialize( method.maximum_order );
        }
    }
    static bool supports( const bdf_flight_method &method )
    {
        return method.maximum_order >= 1 && method.maximum_order <= ( method.adaptive ? 5u : 2u ) &&
               std::isfinite( method.minimum_step_ratio ) && method.minimum_step_ratio > 0 &&
               method.minimum_step_ratio <= 1 && std::isfinite( method.maximum_step_ratio ) &&
               method.maximum_step_ratio >= 1 && method.maximum_step_ratio <= 2;
    }
    static void
    validate( const bdf_flight_method &method, const std::vector<record_type> &steps, std::size_t maximum_steps )
    {
        if ( !supports( method ) )
            throw std::invalid_argument( "Invalid BDF replay method" );
        detail::validate_flight_schedule( steps, maximum_steps );
        unsigned int history_size = 0;
        for ( std::size_t i = 0; i < steps.size(); ++i )
        {
            const auto &r         = steps[i];
            const T     tolerance = T( 4 ) * std::numeric_limits<T>::epsilon() *
                                std::max( { std::abs( r.time ), std::abs( r.candidate_end ), std::abs( r.dt ) } );
            const bool paired =
                i && ( equal_step( r.dt, steps[i - 1].dt ) || std::abs( r.dt - steps[i - 1].dt ) <= tolerance );
            // Honor a recorded restart. In particular, the step following a
            // roundoff-clipped step can restart even though both h are close.
            bool valid_order = r.order == 1 || ( r.order == 2 && method.maximum_order == 2 && paired );
            if ( method.adaptive )
            {
                const T    ratio    = i ? r.dt / steps[i - 1].dt : T( 0 );
                const T    roundoff = T( 8 ) * std::numeric_limits<T>::epsilon();
                const bool restart  = !history_size || !std::isfinite( ratio ) ||
                                     ratio < T( method.minimum_step_ratio ) * ( 1 - roundoff ) ||
                                     ratio > T( method.maximum_step_ratio ) * ( 1 + roundoff );
                if ( restart )
                    history_size = 0;
                valid_order =
                    r.order >= 1 && r.order <= method.maximum_order && ( r.order == 1 || r.order <= history_size );
                history_size = std::min( method.maximum_order, history_size + 1 );
            }
            if ( !valid_order || !r.attempts || !std::isfinite( r.candidate_end ) ||
                 std::abs( r.candidate_end - ( r.time + r.dt ) ) > tolerance ||
                 ( r.theta == T( 1 ) && r.end != r.candidate_end ) ||
                 ( r.dt > 0 ? r.end > r.candidate_end : r.end < r.candidate_end ) )
                throw std::invalid_argument( "Invalid BDF order/history record" );
        }
    }
    void
    run( const bdf_flight_method &method, const std::vector<record_type> &steps, const X &initial, const X *direction,
         X &endpoint, X *derivative, X *time_rate )
    {
        validate( method, steps, maximum_steps_ );
        if ( method.maximum_order != method_.maximum_order || method.adaptive != method_.adaptive ||
             method.minimum_step_ratio != method_.minimum_step_ratio ||
             method.maximum_step_ratio != method_.maximum_step_ratio || bool( direction ) != bool( derivative ) ||
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
        if ( method.adaptive )
        {
            run_adaptive( steps, direction != nullptr, time_rate != nullptr );
            space_->assign( *state_, endpoint );
            if ( derivative )
                space_->assign( *tangent_, *derivative );
            if ( time_rate )
                space_->assign( *time_rate_, *time_rate );
            return;
        }
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
    using history_type = nmfd::time_steppers::bdf::divided_difference_history<Space>;
    std::unique_ptr<history_type> history_, delta_history_;

    // Differentiate the native Newton interpolation basis analytically in time.
    static nmfd::time_steppers::bdf::formula<T>
    dense_derivative( unsigned int order, T theta, T h, const std::array<T, 5> &steps )
    {
        nmfd::time_steppers::bdf::formula<T> c{ order, T( 0 ), T( 0 ) };
        if ( order == 1 )
        {
            c.alpha = T( 1 ) / h;
            return c;
        }
        std::array<T, 5> nodes{};
        for ( unsigned int i = 1; i < order; ++i )
            nodes[i] = nodes[i - 1] - steps[i - 1] / steps[0];
        const T r = h / steps[0], z = theta * r;
        T       weight = 1, derivative = 0;
        for ( unsigned int i = 0; i < order; ++i )
        {
            derivative = ( derivative * ( z - nodes[i] ) + weight ) / ( r - nodes[i] );
            weight *= ( z - nodes[i] ) / ( r - nodes[i] );
        }
        c.alpha  = derivative / steps[0];
        T sample = 1, sample_derivative = 0, endpoint = 1;
        for ( unsigned int i = 0; i + 1 < order; ++i )
        {
            sample_derivative = sample_derivative * ( z - nodes[i] ) + sample;
            sample *= z - nodes[i];
            endpoint *= r - nodes[i];
            c.reference[i] = ( sample_derivative - derivative * endpoint ) / steps[0];
        }
        return c;
    }
    void run_adaptive( const std::vector<record_type> &steps, bool tangent, bool time_rate )
    {
        history_->reset();
        delta_history_->reset();
        std::array<T, 5> previous_steps{};
        for ( std::size_t i = 0; i < steps.size(); ++i )
        {
            const auto &r = steps[i];
            const bool  restart =
                !history_->usable( r.dt, T( method_.minimum_step_ratio ), T( method_.maximum_step_ratio ) );
            T alpha = T( 1 ) / r.dt;
            space_->assign( *state_, *reference_ );
            if ( tangent )
                space_->assign( *tangent_, *delta_reference_ );
            if ( !restart )
            {
                const auto c = history_->coefficients( r.order, r.dt );
                alpha        = c.alpha;
                history_->reference( c, *state_, *reference_ );
                if ( tangent )
                    delta_history_->reference( c, *tangent_, *delta_reference_ );
            }
            space_->assign( *state_, *stage_ );
            if ( !std::isfinite( alpha ) || !provider_->solve_stage( r.candidate_end, alpha, *reference_, *stage_ ) ||
                 !space_->is_valid_number( *stage_ ) )
                throw std::runtime_error( "Adaptive BDF base replay solve failed" );
            if ( tangent && ( !provider_->solve_tangent( *stage_, *reference_, *delta_reference_, *delta_stage_ ) ||
                              !space_->is_valid_number( *delta_stage_ ) ) )
                throw std::runtime_error( "Adaptive BDF tangent solve failed" );
            space_->assign( *stage_, *endpoint_ );
            if ( tangent )
                space_->assign( *delta_stage_, *delta_endpoint_ );
            if ( i + 1 == steps.size() )
            {
                if ( r.theta != T( 1 ) )
                {
                    history_->interpolate( r.order, r.theta, r.dt, *state_, *stage_, *endpoint_ );
                    if ( tangent )
                        delta_history_->interpolate(
                            r.order, r.theta, r.dt, *tangent_, *delta_stage_, *delta_endpoint_
                        );
                }
                if ( time_rate )
                {
                    const auto c = dense_derivative( r.order, r.theta, r.dt, previous_steps );
                    space_->assign_lin_comb( c.alpha, *stage_, -c.alpha, *state_, *time_rate_ );
                    history_->reference( c, *time_rate_, *time_rate_ );
                    if ( !space_->is_valid_number( *time_rate_ ) )
                        throw std::runtime_error( "Nonfinite adaptive BDF dense time derivative" );
                }
            }
            if ( !space_->is_valid_number( *endpoint_ ) || ( tangent && !space_->is_valid_number( *delta_endpoint_ ) ) )
                throw std::runtime_error( "Nonfinite adaptive BDF endpoint" );
            history_->commit( *state_, *stage_, r.dt, restart );
            if ( tangent )
                delta_history_->commit( *tangent_, *delta_stage_, r.dt, restart );
            if ( restart )
                previous_steps.fill( 0 );
            for ( unsigned int j = 4; j > 0; --j )
                previous_steps[j] = previous_steps[j - 1];
            previous_steps[0] = r.dt;
            space_->assign( *endpoint_, *state_ );
            if ( tangent )
                space_->assign( *delta_endpoint_, *tangent_ );
        }
    }
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
        bool         adaptive             = false;
        double       minimum_step_ratio = .2, maximum_step_ratio = 2;
    };
    bdf_flight_factory(
        std::shared_ptr<Space> space, std::shared_ptr<const ProviderFactory> providers, const params &p = {}
    )
        : space_( std::move( space ) ), providers_( std::move( providers ) ), params_( p )
    {
        if ( !space_ || !providers_ || !flow_type::replay_type::supports( method() ) || !p.maximum_replay_steps )
            throw std::invalid_argument( "Invalid BDF flight factory" );
    }
    template <class Segments>
    auto make_workspace( const Segments &segments ) const
    {
        for ( const auto &s : segments )
        {
            if ( !s.flight )
                throw std::invalid_argument( "Missing BDF segment map" );
            flow_type::replay_type::validate( method(), s.flight->records(), params_.maximum_replay_steps );
        }
        return std::make_shared<typename flow_type::replay_type>(
            space_, providers_->make_provider(), method(), params_.maximum_replay_steps
        );
    }
    template <class Map>
    flow_type prepare(
        std::shared_ptr<typename flow_type::replay_type> workspace, const Map &flight,
        const typename Space::vector_type &initial, const typename Space::vector_type &arrival,
        std::shared_ptr<const Section> target, typename Space::scalar_type speed, typename Space::scalar_type tolerance
    ) const
    {
        flow_type::replay_type::validate( method(), flight.records(), params_.maximum_replay_steps );
        return { space_,  std::move( workspace ), method(), flight.records(), initial,
                 arrival, std::move( target ),    speed,    tolerance };
    }

private:
    std::shared_ptr<Space>                 space_;
    std::shared_ptr<const ProviderFactory> providers_;
    params                                 params_;
    bdf_flight_method                      method() const
    {
        return { params_.maximum_order, params_.adaptive, params_.minimum_step_ratio, params_.maximum_step_ratio };
    }
};
}
#endif
