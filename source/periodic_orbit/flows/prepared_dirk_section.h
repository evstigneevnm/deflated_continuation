#ifndef PERIODIC_ORBIT_PREPARED_DIRK_SECTION_H
#define PERIODIC_ORBIT_PREPARED_DIRK_SECTION_H

#include <periodic_orbit/flows/implicit_stage_provider.h>
#include <periodic_orbit/flows/prepared_explicit_section.h>

namespace periodic_orbit
{
template <class Space>
using dirk_stage_context = implicit_stage_context<Space>;

template <class Space, class Problem, class NonlinearSolver, class LinearSolver, class Sensitivity>
using dirk_stage_provider = implicit_stage_provider<Space, Problem, NonlinearSolver, LinearSolver, Sensitivity>;

// Replay solves base stages and their linearizations on the accepted mesh.
// No controller, trajectory-sized vector storage or host staging is involved.
template <class Space, class StageProvider>
class dirk_flight_replay
{
public:
    using T          = typename Space::scalar_type;
    using X          = typename Space::vector_type;
    using table_type = nmfd::time_steppers::runge_kutta::butcher_table;
    using wrap       = nmfd::detail::vector_wrap<Space, false, false>;
    dirk_flight_replay(
        std::shared_ptr<Space> space, std::shared_ptr<StageProvider> provider, std::size_t stages,
        std::size_t maximum_steps = 1000000
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing DIRK replay space" ) ),
          provider_( std::move( provider ) ), maximum_steps_( maximum_steps ), state_( *space_ ), tangent_( *space_ ),
          stage_( *space_ ), delta_stage_( *space_ ), reference_( *space_ ), delta_reference_( *space_ ),
          endpoint_( *space_ ), delta_endpoint_( *space_ )
    {
        if ( !provider_ || !stages || !maximum_steps )
            throw std::invalid_argument( "Empty DIRK replay budget" );
        for ( auto *x :
              { &state_, &tangent_, &stage_, &delta_stage_, &reference_, &delta_reference_, &endpoint_,
                &delta_endpoint_ } )
        {
            x->init();
            x->start_use();
        }
        rates_.reserve( stages );
        tangents_.reserve( stages );
        for ( std::size_t i = 0; i < stages; ++i )
        {
            rates_.emplace_back( *space_ );
            rates_.back().init();
            rates_.back().start_use();
            tangents_.emplace_back( *space_ );
            tangents_.back().init();
            tangents_.back().start_use();
        }
    }
    static bool supports( const table_type &table )
    {
        if ( table.type() == table_type::scheme_type::explicit_rk || !table.has_dense_output() )
            return false;
        const bool stiff = stiffly_accurate( table );
        for ( std::size_t i = 0; i < table.size(); ++i )
        {
            for ( std::size_t j = i + 1; j < table.size(); ++j )
                if ( table.a( i, j ) != 0 )
                    return false;
            if ( table.a( i, i ) == 0 && needs_rate( table, i, true, stiff ) && !StageProvider::supports_rate )
                return false;
        }
        for ( std::size_t i = table.size(); i < table.dense_outout_stage_count(); ++i )
            if ( needs_rate( table, i, true, stiff ) && !StageProvider::supports_rate )
                return false;
        return true;
    }
    void
    run( const table_type &table, const std::vector<flight_step_record<T>> &steps, const X &initial, const X *direction,
         X &endpoint, X *derivative, X *time_rate )
    {
        if ( !supports( table ) || steps.empty() || steps.size() > maximum_steps_ ||
             table.dense_outout_stage_count() > rates_.size() || bool( direction ) != bool( derivative ) ||
             &endpoint == derivative || &endpoint == time_rate || ( derivative && derivative == time_rate ) )
            throw std::invalid_argument( "Invalid DIRK replay configuration" );
        if ( space_->get_size( initial ) != space_->get_vector_size() ||
             space_->get_size( endpoint ) != space_->get_vector_size() ||
             ( direction && ( space_->get_size( *direction ) != space_->get_vector_size() ||
                              space_->get_size( *derivative ) != space_->get_vector_size() ) ) ||
             ( time_rate && space_->get_size( *time_rate ) != space_->get_vector_size() ) )
            throw std::invalid_argument( "DIRK replay vector shape mismatch" );
        space_->assign( initial, *state_ );
        if ( direction )
            space_->assign( *direction, *tangent_ );
        const bool stiff = stiffly_accurate( table );
        for ( std::size_t step = 0; step < steps.size(); ++step )
        {
            const auto &r = steps[step];
            if ( !std::isfinite( r.time ) || !std::isfinite( r.dt ) || r.dt == 0 || !std::isfinite( r.theta ) ||
                 r.theta <= 0 || r.theta > 1 )
                throw std::invalid_argument( "Invalid DIRK replay step" );
            const bool last  = step + 1 == steps.size();
            const auto count = last ? table.dense_outout_stage_count() : table.size();
            for ( std::size_t i = 0; i < count; ++i )
            {
                space_->assign( *state_, *reference_ );
                if ( direction )
                    space_->assign( *tangent_, *delta_reference_ );
                for ( std::size_t j = 0; j < i; ++j )
                    if ( table.dense_outout_a( i, j ) != 0 )
                    {
                        const T weight = r.dt * T( table.dense_outout_a( i, j ) );
                        space_->add_lin_comb( weight, *rates_[j], *reference_ );
                        if ( direction )
                            space_->add_lin_comb( weight, *tangents_[j], *delta_reference_ );
                    }
                space_->assign( *reference_, *stage_ );
                const T    time = r.time + r.dt * T( stiff && i + 1 == table.size() ? 1 : table.dense_outout_c( i ) );
                const T    diagonal = i < table.size() ? T( table.a( i, i ) ) : T( 0 );
                const bool used     = needs_rate( table, i, last, stiff );
                if ( diagonal != 0 )
                {
                    const T alpha = T( 1 ) / ( r.dt * diagonal );
                    if ( !std::isfinite( time ) || !std::isfinite( alpha ) ||
                         !provider_->solve_stage( time, alpha, *reference_, *stage_ ) ||
                         !space_->is_valid_number( *stage_ ) )
                        throw std::runtime_error( "DIRK replay stage solve failed" );
                    if ( direction &&
                         ( !provider_->solve_tangent( *stage_, *reference_, *delta_reference_, *delta_stage_ ) ||
                           !space_->is_valid_number( *delta_stage_ ) ) )
                        throw std::runtime_error( "DIRK replay tangent solve failed" );
                    if ( used )
                    {
                        space_->assign_lin_comb( T( 1 ), *stage_, T( -1 ), *reference_, *rates_[i] );
                        space_->scale( alpha, *rates_[i] );
                        if ( direction )
                        {
                            space_->assign_lin_comb( T( 1 ), *delta_stage_, T( -1 ), *delta_reference_, *tangents_[i] );
                            space_->scale( alpha, *tangents_[i] );
                        }
                    }
                }
                else
                {
                    if ( direction )
                        space_->assign( *delta_reference_, *delta_stage_ );
                    if ( used && ( !provider_->evaluate_rate( time, *stage_, *rates_[i] ) ||
                                   ( direction &&
                                     !provider_->apply_rate_tangent( time, *stage_, *delta_stage_, *tangents_[i] ) ) ) )
                        throw std::runtime_error( "DIRK replay rate derivative unavailable or failed" );
                }
                if ( used && ( !space_->is_valid_number( *rates_[i] ) ||
                               ( direction && !space_->is_valid_number( *tangents_[i] ) ) ) )
                    throw std::runtime_error( "Nonfinite DIRK replay rate" );
                if ( stiff && i + 1 == table.size() )
                {
                    space_->assign( *stage_, *endpoint_ );
                    if ( direction )
                        space_->assign( *delta_stage_, *delta_endpoint_ );
                }
            }
            if ( !stiff || ( last && r.theta != T( 1 ) ) )
            {
                space_->assign( *state_, *endpoint_ );
                if ( direction )
                    space_->assign( *tangent_, *delta_endpoint_ );
                for ( std::size_t i = 0; i < count; ++i )
                {
                    const T weight = last && r.theta != T( 1 ) ? T( table.dense_b( i, r.theta ) )
                                                               : ( i < table.size() ? T( table.b( i ) ) : T( 0 ) );
                    if ( weight != 0 )
                    {
                        space_->add_lin_comb( r.dt * weight, *rates_[i], *endpoint_ );
                        if ( direction )
                            space_->add_lin_comb( r.dt * weight, *tangents_[i], *delta_endpoint_ );
                    }
                }
            }
            if ( !space_->is_valid_number( *endpoint_ ) ||
                 ( direction && !space_->is_valid_number( *delta_endpoint_ ) ) )
                throw std::runtime_error( "Nonfinite DIRK replay endpoint" );
            space_->assign( *endpoint_, *state_ );
            if ( direction )
                space_->assign( *delta_endpoint_, *tangent_ );
            if ( last && time_rate )
            {
                space_->assign_scalar( T( 0 ), *reference_ );
                for ( std::size_t i = 0; i < count; ++i )
                {
                    long double weight = 0;
                    for ( auto j = table.dense_degree(); j > 0; --j )
                        weight = weight * r.theta + j * table.dense_coefficient( i, j - 1 );
                    if ( weight != 0 )
                        space_->add_lin_comb( T( weight ), *rates_[i], *reference_ );
                }
                if ( !space_->is_valid_number( *reference_ ) )
                    throw std::runtime_error( "Nonfinite DIRK dense time derivative" );
            }
        }
        space_->assign( *state_, endpoint );
        if ( derivative )
            space_->assign( *tangent_, *derivative );
        if ( time_rate )
            space_->assign( *reference_, *time_rate );
    }

private:
    std::shared_ptr<Space>         space_;
    std::shared_ptr<StageProvider> provider_;
    std::size_t                    maximum_steps_;
    wrap              state_, tangent_, stage_, delta_stage_, reference_, delta_reference_, endpoint_, delta_endpoint_;
    std::vector<wrap> rates_, tangents_;
    static bool       stiffly_accurate( const table_type &table )
    {
        for ( std::size_t i = 0; i < table.size(); ++i )
            if ( table.a( table.size() - 1, i ) != table.b( i ) )
                return false;
        return true;
    }
    static bool needs_rate( const table_type &table, std::size_t i, bool last, bool stiff )
    {
        if ( !stiff && i < table.size() && table.b( i ) != 0 )
            return true;
        const auto count = last ? table.dense_outout_stage_count() : table.size();
        for ( std::size_t j = i + 1; j < count; ++j )
            if ( table.dense_outout_a( j, i ) != 0 )
                return true;
        if ( last )
            for ( std::size_t j = 0; j < table.dense_degree(); ++j )
                if ( table.dense_coefficient( i, j ) != 0 )
                    return true;
        return false;
    }
};

template <class Space, class Provider, class Section>
using prepared_dirk_section = prepared_rk_section<Space, dirk_flight_replay<Space, Provider>, Section>;

// ProviderFactory::make_provider() creates an isolated mutable stage context for
// each Jacobian. Constructing this factory allocates no tangent vectors/solvers.
template <class Space, class ProviderFactory, class Section>
class dirk_flight_factory
{
public:
    using provider_type = typename ProviderFactory::provider_type;
    using flow_type     = prepared_dirk_section<Space, provider_type, Section>;
    struct params
    {
        std::size_t maximum_replay_steps = 1000000;
    };
    dirk_flight_factory(
        std::shared_ptr<Space> space, std::shared_ptr<const ProviderFactory> providers, const params &p = {}
    )
        : space_( std::move( space ) ), providers_( std::move( providers ) ), params_( p )
    {
        if ( !space_ || !providers_ || !p.maximum_replay_steps )
            throw std::invalid_argument( "Missing DIRK flight factory dependencies" );
    }
    template <class Segments>
    auto make_workspace( const Segments &segments ) const
    {
        for ( const auto &s : segments )
            if ( !s.flight || !flow_type::replay_type::supports( s.flight->table() ) ||
                 s.flight->records().size() > params_.maximum_replay_steps )
                throw std::invalid_argument( "Unsupported or over-budget DIRK replay schedule" );
        return std::make_shared<typename flow_type::replay_type>(
            space_, providers_->make_provider(), maximum_rk_stages( segments ), params_.maximum_replay_steps
        );
    }
    template <class Map>
    flow_type prepare(
        std::shared_ptr<typename flow_type::replay_type> workspace, const Map &flight,
        const typename Space::vector_type &initial, const typename Space::vector_type &arrival,
        std::shared_ptr<const Section> target, typename Space::scalar_type speed, typename Space::scalar_type tolerance
    ) const
    {
        if ( flight.records().size() > params_.maximum_replay_steps )
            throw std::length_error( "DIRK prepared schedule exceeds replay budget" );
        return { space_,  std::move( workspace ), flight.table(), flight.records(), initial,
                 arrival, std::move( target ),    speed,          tolerance };
    }

private:
    std::shared_ptr<Space>                 space_;
    std::shared_ptr<const ProviderFactory> providers_;
    params                                 params_;
};
}
#endif
