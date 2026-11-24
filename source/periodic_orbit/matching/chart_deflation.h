#ifndef PERIODIC_ORBIT_CHART_DEFLATION_H
#define PERIODIC_ORBIT_CHART_DEFLATION_H

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
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
    // No prepared linearization may be reused after changing its multiplier.
    void invalidate_linearization() { base::set_parameter( this->parameter() ); }
    void roots_changed() { invalidate_linearization(); }
};

enum class chart_root_status { ready, rechart_failed, matching_failed, period_mismatch, capacity_reached };

template <class Operator>
struct chart_deflation_result
{
    std::shared_ptr<Operator> op;
    chart_root_status status;
    std::optional<std::size_t> failed_root;
};

// Own the backend workspaces for one layout. Reserve handles up front so
// appending a root never relocates numerical data for previous roots.
template <class Space, class Orbit>
class chart_root_set
{
public:
    using T = typename Space::scalar_type;
    using operator_type = chart_deflated_operator<Space, Orbit>;
    chart_root_set( std::shared_ptr<Space> coords, std::shared_ptr<Orbit> orbit, T parameter,
                    std::size_t capacity, T matching_tol, T period_tol, T exclusion )
        : coords_( std::move( coords ) ), orbit_( std::move( orbit ) ), matching_tol_( matching_tol ),
          period_tol_( period_tol ), stored_( make_storage( capacity, parameter, exclusion ) ),
          op_( std::make_shared<operator_type>( coords_, orbit_, orbit_, stored_, parameter, exclusion ) ),
          z_( *coords_ ), residual_( *coords_ )
    {}
    std::size_t size() const { return stored_->get_size(); }
    const std::shared_ptr<operator_type> &op() const { return op_; }

    template <class Crossing, class Rechart>
    chart_root_status append( const Crossing &root, Rechart &rechart )
    {
        if ( size() == stored_->get_number_of_reserved_solutions() ) return chart_root_status::capacity_reached;
        return with_coordinates( root, rechart, [&]( const auto &z ) {
            stored_->push_back( z );
            op_->roots_changed();
        } );
    }

    // Borrow the checked coordinates for the synchronous callback only.
    template <class Crossing, class Rechart, class Use>
    chart_root_status with_coordinates( const Crossing &root, Rechart &rechart, Use &&use )
    {
        if ( !rechart( root.state(), root.period(), *z_ ) ) return chart_root_status::rechart_failed;
        const auto &r = orbit_->inspect_crossings( *z_, op_->parameter(), *residual_ );
        if ( !r.succeeded() || !(coords_->norm( *residual_ ) <= matching_tol_) )
            return chart_root_status::matching_failed;
        if ( !(common::scalar_math::abs( r.period - root.period() ) <= period_tol_) )
            return chart_root_status::period_mismatch;
        use( std::as_const( *z_ ) );
        return chart_root_status::ready;
    }
private:
    using storage = deflation::solution_storage<Space>;
    std::shared_ptr<storage> make_storage( std::size_t capacity, T parameter, T exclusion ) const
    {
        using common::scalar_math::isfinite;
        if ( !coords_ || !orbit_ || !coords_->component_count() || !isfinite( parameter ) ||
             !isfinite( matching_tol_ ) || matching_tol_ <= 0 || !isfinite( period_tol_ ) || period_tol_ <= 0 ||
             !isfinite( exclusion ) || exclusion < 0 || capacity > std::numeric_limits<unsigned>::max() )
            throw std::invalid_argument( "Invalid fixed-chart deflation configuration" );
        // Glued norm_l2^2 sums the component RMS squares, not the global RMS.
        const auto n = coords_->space_comp( 0 ).get_vector_size();
        for ( std::size_t i = 0; i < coords_->component_count(); ++i )
            if ( !n || coords_->space_comp( i ).get_vector_size() != n )
                throw std::invalid_argument( "Expected homogeneous orbit chart dimensions" );
        auto s = std::make_shared<storage>( coords_.get(), static_cast<unsigned>( capacity ), T(n) );
        s->set_ignore_zero();
        return s;
    }
    std::shared_ptr<Space> coords_;
    std::shared_ptr<Orbit> orbit_;
    T matching_tol_, period_tol_;
    std::shared_ptr<storage> stored_;
    std::shared_ptr<operator_type> op_;
    nmfd::detail::vector_wrap<Space, true, true> z_, residual_;
};

// Full preparation is transactional; an incomplete operator never escapes.
template <class Collection, class Space, class Orbit, class Rechart>
auto prepare_chart_deflation( const Collection &roots, const std::shared_ptr<Space> &coords,
                              const std::shared_ptr<Orbit> &orbit, Rechart rechart,
                              typename Space::scalar_type matching_tol, typename Space::scalar_type period_tol,
                              typename Space::scalar_type exclusion )
{
    using result = chart_deflation_result<chart_deflated_operator<Space, Orbit>>;
    chart_root_set<Space, Orbit> chart( coords, orbit, roots.parameter(), roots.size(), matching_tol, period_tol, exclusion );
    for ( std::size_t i = 0; i < roots.size(); ++i )
    {
        const auto status = chart.append( roots.at( i ), rechart );
        if ( status != chart_root_status::ready ) return result{ nullptr, status, i };
    }
    return result{ chart.op(), chart_root_status::ready, {} };
}
}
#endif
