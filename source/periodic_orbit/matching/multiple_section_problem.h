#ifndef PERIODIC_ORBIT_MULTIPLE_SECTION_PROBLEM_H
#define PERIODIC_ORBIT_MULTIPLE_SECTION_PROBLEM_H

#include <periodic_orbit/flows/prepared_explicit_section.h>
#include <periodic_orbit/matching/multiple_section_residual.h>

namespace periodic_orbit
{
template <
    class Space, class CoordinateSpace, class RecordedMap, class RateProvider,
    class Chart = affine_chart<Space, CoordinateSpace>>
class multiple_section_problem
{
public:
    using residual_type = multiple_section_residual<Space, CoordinateSpace, RecordedMap, Chart>;
    using space_type    = typename residual_type::space_type;
    using vector_type   = typename residual_type::vector_type;
    using T             = typename Space::scalar_type;
    using layout_type   = typename residual_type::layout_type;
    using segment       = typename residual_type::segment;
    using params        = typename residual_type::params;
    using flow_type     = prepared_explicit_section<Space, RateProvider, typename Chart::section_type>;
    using physical_wrap = nmfd::detail::vector_wrap<Space, false, false>;
    using block_wrap    = nmfd::detail::vector_wrap<space_type, false, false>;

    class jacobi_operator_type
    {
    public:
        jacobi_operator_type(
            std::shared_ptr<Space> physical, std::shared_ptr<space_type> coordinates, const layout_type &layout,
            const std::vector<segment> &segments, std::shared_ptr<const RateProvider> rate, const vector_type &base,
            const vector_type &residual, const params &p
        )
            : physical_( std::move( physical ) ), coordinates_( std::move( coordinates ) ), pending_( *coordinates_ ),
              direction_( *physical_ ), image_( *physical_ )
        {
            std::size_t stages = 0;
            for ( const auto &s : segments )
                stages = std::max( stages, s.flight->table().dense_outout_stage_count() );
            auto replay = std::make_shared<typename flow_type::replay_type>( physical_, std::move( rate ), stages );
            pending_.init();
            pending_.start_use();
            direction_.init();
            direction_.start_use();
            image_.init();
            image_.start_use();
            flows_.reserve( segments.size() );
            charts_.reserve( segments.size() );
            time_increments_.resize( segments.size() );
            pending_times_.resize( segments.size() );
            for ( std::size_t i = 0; i < segments.size(); ++i )
            {
                const auto next = ( i + 1 ) % segments.size();
                charts_.push_back( segments[i].chart );
                segments[i].chart->lift( base.comp( i ), *direction_ );
                // The residual retains coordinates, so reconstruct the located
                // arrival within the checked physical section tolerance.
                coordinates_->space_comp( i ).assign_lin_comb(
                    1, base.comp( next ), 1, residual.comp( i ), ( *pending_ ).comp( i )
                );
                segments[next].chart->lift( ( *pending_ ).comp( i ), *image_ );
                flows_.emplace_back(
                    physical_, replay, segments[i].flight->table(), segments[i].flight->records(), *direction_, *image_,
                    layout.at( next ).section, p.rule.minimum_crossing_speed, p.section_tolerance * T( 10 )
                );
            }
        }
        // A serial operator owns its scratch; in/out may alias. No mesh or
        // controller decisions occur during this matrix-free linear action.
        void apply( const vector_type &in, vector_type &out ) const
        {
            if ( in.size() != flows_.size() || out.size() != flows_.size() )
                throw std::invalid_argument( "Matching Jacobian block count mismatch" );
            for ( std::size_t i = 0; i < flows_.size(); ++i )
                if ( coordinates_->space_comp( i ).get_size( in.comp( i ) ) !=
                         coordinates_->space_comp( i ).get_vector_size() ||
                     coordinates_->space_comp( i ).get_size( out.comp( i ) ) !=
                         coordinates_->space_comp( i ).get_vector_size() )
                    throw std::invalid_argument( "Matching Jacobian component shape mismatch" );
            if ( !coordinates_->is_valid_number( in ) )
                throw std::runtime_error( "Nonfinite matching direction" );
            for ( std::size_t i = 0; i < flows_.size(); ++i )
            {
                const auto next = ( i + 1 ) % flows_.size();
                charts_[i]->lift_increment( in.comp( i ), *direction_ );
                flows_[i].apply( *direction_, *image_ );
                pending_times_[i] = flows_[i].last_time_increment();
                charts_[next]->restrict_increment( *image_, ( *pending_ ).comp( i ) );
                coordinates_->space_comp( i ).add_lin_comb( -1, in.comp( next ), ( *pending_ ).comp( i ) );
            }
            if ( !coordinates_->is_valid_number( *pending_ ) )
                throw std::runtime_error( "Nonfinite matching Jacobian" );
            coordinates_->assign( *pending_, out );
            std::copy( pending_times_.begin(), pending_times_.end(), time_increments_.begin() );
        }
        T flight_time_increment( std::size_t i ) const
        {
            return time_increments_.at( i );
        }

    private:
        std::shared_ptr<Space>              physical_;
        std::shared_ptr<space_type>         coordinates_;
        std::vector<std::shared_ptr<Chart>> charts_;
        mutable std::vector<T>              pending_times_, time_increments_;
        std::vector<flow_type>              flows_;
        mutable block_wrap                  pending_;
        mutable physical_wrap               direction_, image_;
    };

    multiple_section_problem(
        std::shared_ptr<Space> physical, std::shared_ptr<space_type> coordinates,
        std::shared_ptr<const layout_type> layout, std::vector<segment> segments,
        std::shared_ptr<const RateProvider> rate, const params &p = {}
    )
        : physical_( std::move( physical ) ), coordinates_( std::move( coordinates ) ), layout_( std::move( layout ) ),
          segments_( std::move( segments ) ),
          rate_( rate ? std::move( rate ) : throw std::invalid_argument( "Missing tangent rate provider" ) ),
          params_( p ), residual_( physical_, coordinates_, layout_, segments_, p ), scratch_( *coordinates_ )
    {
        scratch_.init();
        scratch_.start_use();
    }
    void apply( const vector_type &in, vector_type &out )
    {
        residual_.apply( in, out );
    }
    const auto &report() const
    {
        return residual_.report();
    }
    void set_linearization_point( const vector_type &base )
    {
        // Nothing is published until all flights and replay checks succeed.
        residual_.apply( base, *scratch_ );
        auto pending = std::make_shared<jacobi_operator_type>(
            physical_, coordinates_, *layout_, segments_, rate_, base, *scratch_, params_
        );
        jacobi_ = std::move( pending );
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        if ( !jacobi_ )
            throw std::logic_error( "Prepare the matching Jacobian first" );
        return jacobi_;
    }

private:
    std::shared_ptr<Space>                      physical_;
    std::shared_ptr<space_type>                 coordinates_;
    std::shared_ptr<const layout_type>          layout_;
    std::vector<segment>                        segments_;
    std::shared_ptr<const RateProvider>         rate_;
    params                                      params_;
    residual_type                               residual_;
    block_wrap                                  scratch_;
    std::shared_ptr<const jacobi_operator_type> jacobi_;
};
}
#endif
