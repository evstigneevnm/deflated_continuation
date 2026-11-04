#ifndef PERIODIC_ORBIT_MULTIPLE_SECTION_RESIDUAL_H
#define PERIODIC_ORBIT_MULTIPLE_SECTION_RESIDUAL_H

#include <algorithm>
#include <nmfd/operations/glued_vector_space.h>
#include <periodic_orbit/sections/affine_chart.h>
#include <periodic_orbit/sections/section_layout.h>

namespace periodic_orbit
{
template <class T>
struct matching_segment_report
{
    map_report<T> flight;
    T             initial_section_value{}, arrival_section_value{}, gap_norm{};
};

template <class T>
struct matching_report
{
    section_status                          status         = section_status::idle;
    std::size_t                             failed_section = 0;
    T                                       period{};
    std::vector<matching_segment_report<T>> segments;
    bool                                    succeeded() const
    {
        return status == section_status::located;
    }
};

// Expected flight failures cross the void nonlinear-residual boundary explicitly.
// Callers solve into scratch and decide whether/how to retry the failed attempt.
template <class T>
class matching_evaluation_error : public std::runtime_error
{
public:
    section_status status;
    std::size_t    section;
    map_report<T>  flight;
    explicit matching_evaluation_error( const matching_report<T> &report )
        : std::runtime_error( "Multiple-section matching evaluation failed" ), status( report.status ),
          section( report.failed_section ), flight( report.segments.at( section ).flight )
    {
    }
};

// Autonomous, transverse PO matching in fixed section charts. No Jacobian is
// advertised until a prepared numerical flight derivative is available.
template <class Space, class CoordinateSpace, class Map, class Chart = affine_chart<Space, CoordinateSpace>>
class multiple_section_residual
{
public:
    using T            = typename Space::norm_type;
    using X            = typename Space::vector_type;
    using space_type   = nmfd::operations::glued_vector_space<CoordinateSpace>;
    using vector_type  = typename space_type::vector_type;
    using section_type = typename Chart::section_type;
    using layout_type  = section_layout<section_type>;
    using report_type  = matching_report<T>;
    using traits       = scfd::utils::scalar_traits<T>;
    struct segment
    {
        std::shared_ptr<Map>   flight;
        std::shared_ptr<Chart> chart;
    };
    struct params
    {
        T                flight_limit_factor = T( 2 );
        T                section_tolerance   = T( 1e-8 );
        crossing_rule<T> rule;
        params()
        {
            rule.minimum_crossing_speed = T( 1e-10 );
        }
    };

    multiple_section_residual(
        std::shared_ptr<Space> physical, std::shared_ptr<space_type> coordinates,
        std::shared_ptr<const layout_type> layout, std::vector<segment> segments, const params &p = {}
    )
        : physical_( physical ? std::move( physical ) : throw std::invalid_argument( "Missing matching space" ) ),
          coordinates_(
              coordinates ? std::move( coordinates ) : throw std::invalid_argument( "Missing matching coordinates" )
          ),
          layout_( layout ? std::move( layout ) : throw std::invalid_argument( "Missing matching layout" ) ),
          segments_( std::move( segments ) ), params_( p ), pending_( *coordinates_ ), initial_( *physical_ ),
          arrival_( *physical_ ), section_work_( physical_ )
    {
        p.rule.validate( T( 0 ) );
        if ( segments_.size() != layout_->size() || coordinates_->component_count() != layout_->size() ||
             !traits::is_finite( p.flight_limit_factor ) || p.flight_limit_factor <= T( 1 ) ||
             !traits::is_finite( p.section_tolerance ) || p.section_tolerance <= T( 0 ) ||
             p.rule.initial != initial_event_policy::leave_and_return || p.rule.minimum_crossing_speed <= T( 0 ) )
            throw std::invalid_argument( "Invalid multiple-section matching configuration" );
        if ( layout_->closing_target().get() != layout_->at( 0 ).section.get() )
            throw std::invalid_argument( "Transformed closure requires a relative-orbit matching policy" );
        for ( std::size_t i = 0; i < segments_.size(); ++i )
        {
            const auto &node = layout_->at( i );
            if ( !segments_[i].flight || !segments_[i].chart || !physical_->get_vector_size() ||
                 coordinates_->space_comp( i ).get_vector_size() != physical_->get_vector_size() - 1 ||
                 ( node.expected_flight > T( 0 ) ) != ( layout_->at( 0 ).expected_flight > T( 0 ) ) ||
                 !traits::is_finite( node.pilot_time + p.flight_limit_factor * node.expected_flight ) )
                throw std::invalid_argument( "Invalid matching segment" );
        }
        report_.segments.resize( segments_.size() );
        pending_.init();
        pending_.start_use();
        initial_.init();
        initial_.start_use();
        arrival_.init();
        arrival_.start_use();
    }

    // Reports are borrowed until the next evaluation. No component is published
    // before every flight and geometry check succeeds; in/out may alias.
    const report_type &evaluate( const vector_type &in, vector_type &out )
    {
        report_.status         = section_status::invalid_input;
        report_.period         = T( 0 );
        report_.failed_section = 0;
        std::fill( report_.segments.begin(), report_.segments.end(), matching_segment_report<T>{} );
        if ( !valid_shape( in ) || !valid_shape( out ) )
            return report_;
        if ( !coordinates_->is_valid_number( in ) )
            return fail( section_status::invalid_state );
        for ( std::size_t i = 0; i < segments_.size(); ++i )
        {
            report_.failed_section = i;
            const auto  next       = ( i + 1 ) % segments_.size();
            auto       &diagnostic = report_.segments[i];
            const auto &node       = layout_->at( i );
            segments_[i].chart->lift( in.comp( i ), *initial_ );
            diagnostic.initial_section_value = node.section->bind( section_work_ ).value( node.pilot_time, *initial_ );
            if ( !physical_->is_valid_number( *initial_ ) || !on_section( diagnostic.initial_section_value ) )
                return fail( section_status::invalid_state );
            flight_request<T, section_type> request;
            request.start_time = node.pilot_time;
            request.end_time   = node.pilot_time + params_.flight_limit_factor * node.expected_flight;
            request.target     = layout_->at( next ).section;
            request.rule       = params_.rule;
            report_.status     = section_status::step_failure;
            diagnostic.flight  = segments_[i].flight->evaluate( *initial_, request, *arrival_ );
            const auto &flight = diagnostic.flight;
            if ( !flight.succeeded() )
                return fail( flight.status );
            if ( !traits::is_finite( flight.time ) || !traits::is_finite( flight.flight_time ) ||
                 flight.flight_time == T( 0 ) || ( flight.flight_time > T( 0 ) ) != ( node.expected_flight > T( 0 ) ) ||
                 !physical_->is_valid_number( *arrival_ ) )
                return fail( section_status::invalid_state );
            diagnostic.arrival_section_value = request.target->bind( section_work_ ).value( flight.time, *arrival_ );
            if ( !on_section( diagnostic.arrival_section_value ) )
                return fail( section_status::invalid_state );
            segments_[next].chart->restrict( *arrival_, ( *pending_ ).comp( i ) );
            coordinates_->space_comp( i ).add_lin_comb( -1, in.comp( next ), ( *pending_ ).comp( i ) );
            // Verify the physical gap too: chart restriction must not conceal a
            // mismatched target plane or silently discard its normal component.
            segments_[next].chart->lift( in.comp( next ), *initial_ );
            physical_->add_lin_comb( -1, *arrival_, *initial_ );
            diagnostic.gap_norm = physical_->norm( *initial_ );
            report_.period += flight.flight_time;
            if ( !traits::is_finite( diagnostic.gap_norm ) || !traits::is_finite( report_.period ) )
                return fail( section_status::invalid_state );
        }
        if ( !coordinates_->is_valid_number( *pending_ ) )
            return fail( section_status::invalid_state );
        coordinates_->assign( *pending_, out );
        report_.status = section_status::located;
        return report_;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        if ( !evaluate( in, out ).succeeded() )
            throw matching_evaluation_error<T>( report_ );
    }
    const report_type &report() const
    {
        return report_;
    }

private:
    bool valid_shape( const vector_type &x ) const
    {
        if ( x.size() != segments_.size() )
            return false;
        for ( std::size_t i = 0; i < segments_.size(); ++i )
            if ( coordinates_->space_comp( i ).get_size( x.comp( i ) ) !=
                 coordinates_->space_comp( i ).get_vector_size() )
                return false;
        return true;
    }
    bool on_section( T value ) const
    {
        return traits::is_finite( value ) && traits::abs( value ) <= params_.section_tolerance;
    }
    const report_type &fail( section_status status )
    {
        report_.status = status;
        return report_;
    }

    std::shared_ptr<Space>                              physical_;
    std::shared_ptr<space_type>                         coordinates_;
    std::shared_ptr<const layout_type>                  layout_;
    std::vector<segment>                                segments_;
    params                                              params_;
    report_type                                         report_;
    nmfd::detail::vector_wrap<space_type, false, false> pending_;
    nmfd::detail::vector_wrap<Space, false, false>      initial_, arrival_;
    typename section_type::evaluation_workspace         section_work_;
};
}
#endif
