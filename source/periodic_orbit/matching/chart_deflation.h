#ifndef PERIODIC_ORBIT_CHART_DEFLATION_H
#define PERIODIC_ORBIT_CHART_DEFLATION_H

#include <optional>
#include <deflation/deflated_operator.h>
#include <deflation/solution_storage.h>

namespace periodic_orbit
{
// Same rank-one operator as stationary deflation, but its roots are bound to
// one parameter and frozen shooting layout. Rebuild rather than rebind.
template <class Space, class Orbit>
class chart_deflated_operator : public deflation::deflated_operator<Space, Orbit, Orbit, deflation::solution_storage<Space>>
{
    using base = deflation::deflated_operator<Space, Orbit, Orbit, deflation::solution_storage<Space>>;
    using base::set_parameter;
public:
    using base::base;
};

enum class chart_root_status { ready, rechart_failed, matching_failed, period_mismatch };

template <class Operator>
struct chart_deflation_result
{
    std::shared_ptr<Operator> op;
    chart_root_status status;
    std::optional<std::size_t> failed_root;
};

// Rechart(physical_crossing, period, coordinates) replays the stored orbit
// into the current layout. Publish no partially populated operator on failure.
template <class Collection, class Space, class Orbit, class Rechart>
auto prepare_chart_deflation( const Collection &roots, const std::shared_ptr<Space> &coords,
                              const std::shared_ptr<Orbit> &orbit, Rechart rechart,
                              typename Space::scalar_type matching_tol, typename Space::scalar_type period_tol,
                              typename Space::scalar_type exclusion )
{
    using T = typename Space::scalar_type;
    using op = chart_deflated_operator<Space, Orbit>;
    using result = chart_deflation_result<op>;
    using common::scalar_math::isfinite;
    if ( !coords || !orbit || !coords->component_count() || !isfinite( matching_tol ) || matching_tol <= 0 ||
         !isfinite( period_tol ) || period_tol <= 0 || !isfinite( exclusion ) || exclusion < 0 ||
         roots.size() > std::numeric_limits<unsigned>::max() )
        throw std::invalid_argument( "Invalid fixed-chart deflation configuration" );
    // Glued norm_l2^2 sums the component RMS squares, not the global RMS.
    const auto n = coords->space_comp( 0 ).get_vector_size();
    for ( std::size_t i = 0; i < coords->component_count(); ++i )
        if ( !n || coords->space_comp( i ).get_vector_size() != n )
            throw std::invalid_argument( "Expected homogeneous orbit chart dimensions" );
    auto stored = std::make_shared<deflation::solution_storage<Space>>( coords.get(), unsigned( roots.size() ), T(n) );
    stored->set_ignore_zero();
    nmfd::detail::vector_wrap<Space, true, true> z( *coords ), residual( *coords );
    for ( std::size_t i = 0; i < roots.size(); ++i )
    {
        const auto &root = roots.at( i );
        if ( !rechart( root.state(), root.period(), *z ) )
            return result{ nullptr, chart_root_status::rechart_failed, i };
        const auto &r = orbit->inspect_crossings( *z, roots.parameter(), *residual );
        if ( !r.succeeded() || !(coords->norm( *residual ) <= matching_tol) )
            return result{ nullptr, chart_root_status::matching_failed, i };
        if ( !(common::scalar_math::abs( r.period - root.period() ) <= period_tol) )
            return result{ nullptr, chart_root_status::period_mismatch, i };
        stored->push_back( *z );
    }
    return result{ std::make_shared<op>( coords, orbit, orbit, stored, roots.parameter(), exclusion ),
                   chart_root_status::ready, {} };
}
}
#endif
