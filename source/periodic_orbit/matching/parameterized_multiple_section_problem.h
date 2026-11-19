#ifndef PERIODIC_ORBIT_PARAMETERIZED_MULTIPLE_SECTION_PROBLEM_H
#define PERIODIC_ORBIT_PARAMETERIZED_MULTIPLE_SECTION_PROBLEM_H

#include <periodic_orbit/matching/multiple_section_problem.h>

namespace periodic_orbit
{
// Binding updates trial flight evaluators in place, but returns a replay factory
// with an independent, fixed parameter for each Newton linearization.
template <class Matching, class Binding>
class parameterized_multiple_section_problem
{
public:
    using scalar_type                   = typename Matching::T;
    using vector_type                   = typename Matching::vector_type;
    using space_type                    = typename Matching::space_type;
    using jacobi_operator_type          = typename Matching::jacobi_operator_type;
    using is_periodic_orbit_reprojected = std::false_type;

    parameterized_multiple_section_problem(
        std::shared_ptr<Matching> matching, Binding binding, std::shared_ptr<space_type> space
    )
        : matching_( matching ? std::move( matching ) : throw std::invalid_argument( "Missing orbit matcher" ) ),
          binding_( std::move( binding ) ),
          space_( space ? std::move( space ) : throw std::invalid_argument( "Missing orbit space" ) ),
          column_( *space_ )
    {
        column_.init();
        column_.start_use();
    }
    void F( const vector_type &z, scalar_type lambda, vector_type &out )
    {
        binding_.set_trial_parameter( lambda );
        matching_->apply( z, out );
    }
    const auto &report() const
    {
        return matching_->report();
    }
    const auto &inspect_crossings( const vector_type &z, scalar_type lambda, vector_type &out )
    {
        binding_.set_trial_parameter( lambda );
        return matching_->inspect_crossings( z, out );
    }
    void set_linearization_point( const vector_type &z, scalar_type lambda )
    {
        binding_.set_trial_parameter( lambda );
        auto next       = matching_->prepare( z, binding_.make_factory( lambda ) );
        period_partial_ = next->parameter_column( *column_ ); // Publishes after every flight succeeds.
        jacobian_       = std::move( next );
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        prepared();
        return jacobian_;
    }
    void jacobian_alpha( vector_type &out ) const
    {
        prepared();
        space_->assign( *column_, out );
    }
    scalar_type period_parameter_partial() const
    {
        prepared();
        return period_partial_;
    }
    // Stable state-block handle for the project's bordered corrector.
    void apply( const vector_type &in, vector_type &out ) const
    {
        prepared().apply( in, out );
    }

    struct return_operator
    {
        using vector_type = typename Matching::coordinate_vector;
        std::shared_ptr<const jacobi_operator_type> preparation;
        void                                        apply( const vector_type &in, vector_type &out ) const
        {
            preparation->apply_return_derivative( in, out );
        }
    };
    return_operator make_return_operator() const
    {
        return { get_jacobi_operator() };
    }

private:
    const jacobi_operator_type &prepared() const
    {
        if ( !jacobian_ )
            throw std::logic_error( "Prepare the orbit linearization first" );
        return *jacobian_;
    }
    std::shared_ptr<Matching>                   matching_;
    Binding                                     binding_;
    std::shared_ptr<space_type>                 space_;
    typename Matching::block_wrap               column_;
    scalar_type                                 period_partial_{};
    std::shared_ptr<const jacobi_operator_type> jacobian_;
};

// The legacy continuation vocabulary uses a rank-one norm on (z, lambda).
// Use precisely the same dot product as the pseudo-arclength constraint.
template <class GluedSpace>
struct orbit_continuation_space : GluedSpace
{
    using GluedSpace::GluedSpace;
    using T = typename GluedSpace::scalar_type;
    using X = typename GluedSpace::vector_type;
    T norm_rank1( const X &z, T lambda ) const
    {
        using std::sqrt;
        return sqrt( this->scalar_prod( z, z ) + lambda * lambda );
    }
    T get_l2_size() const
    {
        using std::sqrt;
        return sqrt( T( this->get_vector_size() ) );
    }
};

// Adapt numerical flight failures at the attempt boundary, not inside Krylov.
struct orbit_failure
{
    section_status status  = section_status::idle;
    std::size_t    section = 0;
    bool           layout_related() const
    {
        return status == section_status::grazing || status == section_status::flight_limit ||
               status == section_status::not_found;
    }
};
template <class Scalar, class Attempt>
bool try_orbit_solve( Attempt attempt, orbit_failure *failure = nullptr )
{
    if ( failure )
        *failure = {};
    try
    {
        return attempt();
    }
    catch ( const matching_evaluation_error<Scalar> &error )
    {
        if ( error.status == section_status::invalid_input )
            throw;
        if ( failure )
            *failure = { error.status, error.section };
        return false;
    }
    catch ( const grazing_section_error & )
    {
        if ( failure )
            failure->status = section_status::grazing;
        return false;
    }
}
template <class Correction>
class orbit_correction : public Correction
{
public:
    using Correction::Correction;
    orbit_failure last_failure;
    template <class... Args>
    bool solve( Args &&...args )
    {
        return try_orbit_solve<typename Correction::scalar_type>(
            [&] { return Correction::solve( std::forward<Args>( args )... ); }, &last_failure
        );
    }
};
template <class Tangent>
class orbit_tangent : public Tangent
{
public:
    using Tangent::Tangent;
    template <class... Args>
    bool initial( Args &&...args )
    {
        return try_orbit_solve<typename Tangent::scalar_type>( [&] {
            return Tangent::initial( std::forward<Args>( args )... );
        } );
    }
    template <class... Args>
    bool apply( Args &&...args )
    {
        return try_orbit_solve<typename Tangent::scalar_type>( [&] {
            return Tangent::apply( std::forward<Args>( args )... );
        } );
    }
};
}
#endif
