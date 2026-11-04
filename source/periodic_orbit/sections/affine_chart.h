#ifndef PERIODIC_ORBIT_AFFINE_CHART_H
#define PERIODIC_ORBIT_AFFINE_CHART_H

#include <type_traits>
#include <periodic_orbit/sections/affine_section.h>
#include <periodic_orbit/sections/detail/scfd_chart_transport.h>

namespace periodic_orbit
{
// Real Euclidean section coordinates. Geometry is immutable, but this chart owns
// scratch: construct independent charts/workspaces for concurrent consumers.
template <class Space, class CoordinateSpace = Space, class Transport = scfd_chart_transport<Space, CoordinateSpace>>
class affine_chart
{
public:
    using scalar_type     = typename Space::scalar_type;
    using vector_type     = typename Space::vector_type;
    using coordinate_type = typename CoordinateSpace::vector_type;
    using section_type    = affine_section<Space>;
    using T               = scalar_type;
    using X               = vector_type;
    using Z               = coordinate_type;
    using traits          = scfd::utils::scalar_traits<T>;
    static_assert(
        std::is_same_v<T, typename Space::norm_type> && std::is_same_v<T, typename CoordinateSpace::scalar_type>,
        "The built-in affine chart requires real scalar coordinates"
    );

    affine_chart(
        std::shared_ptr<Space> space, std::shared_ptr<CoordinateSpace> coordinates,
        std::shared_ptr<const section_type> section, Transport transport = {}
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing chart state space" ) ),
          coordinates_(
              coordinates ? std::move( coordinates ) : throw std::invalid_argument( "Missing chart coordinate space" )
          ),
          section_( section ? std::move( section ) : throw std::invalid_argument( "Missing chart section" ) ),
          transport_( std::move( transport ) ), reflector_( *space_ ), scratch_( *space_ )
    {
        const auto count = space_->get_vector_size();
        if ( !count || coordinates_->get_vector_size() != count - 1 ||
             space_->get_size( section_->center() ) != count || space_->get_size( section_->normal() ) != count )
            throw std::invalid_argument( "Affine chart dimension mismatch" );
        reflector_.init();
        reflector_.start_use();
        scratch_.init();
        scratch_.start_use();
        transport_.make_reflector( section_->normal(), *reflector_ );
        const T squared = space_->norm_sq( *reflector_ );
        if ( !traits::is_finite( squared ) || squared <= T( 0 ) )
            throw std::invalid_argument( "Invalid affine chart reflector" );
        beta_ = T( 2 ) / squared;
    }

    // x = center + H*[0,z], with H an orthogonal Householder reflector.
    void lift( const Z &z, X &state ) const
    {
        lift_increment( z, state );
        space_->add_lin_comb( T( 1 ), section_->center(), state );
    }
    // Off-section inputs are orthogonally projected; a matching workflow must
    // still check its physical section/event residual independently.
    void restrict( const X &state, Z &z ) const
    {
        check_dimensions( state, z );
        space_->assign_lin_comb( T( 1 ), state, T( -1 ), section_->center(), *scratch_ );
        reflect();
        transport_.restrict( *scratch_, z );
    }
    void lift_increment( const Z &v, X &delta ) const
    {
        check_dimensions( delta, v );
        transport_.embed( v, *scratch_ );
        reflect();
        space_->assign( *scratch_, delta );
    }
    void restrict_increment( const X &delta, Z &v ) const
    {
        check_dimensions( delta, v );
        space_->assign( delta, *scratch_ );
        reflect();
        transport_.restrict( *scratch_, v );
    }

private:
    void check_dimensions( const X &state, const Z &z ) const
    {
        if ( space_->get_size( state ) != space_->get_vector_size() ||
             coordinates_->get_size( z ) != coordinates_->get_vector_size() )
            throw std::invalid_argument( "Affine chart vector dimension mismatch" );
    }
    void reflect() const
    {
        const T weight = -beta_ * space_->scalar_prod( *reflector_, *scratch_ );
        space_->add_lin_comb( weight, *reflector_, *scratch_ );
    }
    std::shared_ptr<Space>                                 space_;
    std::shared_ptr<CoordinateSpace>                       coordinates_;
    std::shared_ptr<const section_type>                    section_;
    Transport                                              transport_;
    nmfd::detail::vector_wrap<Space, false, false>         reflector_;
    mutable nmfd::detail::vector_wrap<Space, false, false> scratch_;
    T                                                      beta_{};
};
}
#endif
