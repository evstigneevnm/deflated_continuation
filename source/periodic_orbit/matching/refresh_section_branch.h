#ifndef PERIODIC_ORBIT_REFRESH_SECTION_BRANCH_H
#define PERIODIC_ORBIT_REFRESH_SECTION_BRANCH_H

#include <cmath>
#include <stdexcept>
#include <nmfd/detail/vector_wrap.h>
#include <continuation/tangent_normalization.h>

namespace periodic_orbit
{
// Build independent flights/solvers on the same backend spaces. No accepted
// vector is written until both orbits and the tangent pass in the new charts.
template <class Setup, class State, class T, class Build, class Verify, class Rephase>
auto refresh_section_branch(
    const Setup &old, State &z, T p, State &previous, T prev_p, T prev_period, State &v, T &vp, T tolerance,
    Build build, Verify verify, Rephase rephase
)
{
    using physical_space = typename decltype( old.physical )::element_type;
    nmfd::detail::vector_wrap<physical_space, true, true> seed( *old.physical ), prior( *old.physical );
    old.first->lift( z.comp( 0 ), *seed );
    auto  proposed = build( *seed );
    auto &f        = proposed->flights;
    auto  root = f.workspace(), before = f.workspace(), row = f.workspace(), tangent = f.workspace();
    f.coords->assign_scalar( 0, *root );
    if ( !proposed->solver.ordinary.solve( f.op.get(), *root, p ) || !verify( f, *root ) )
        throw std::runtime_error( "Replacement layout did not reproduce the accepted orbit" );
    f.first->lift( ( *root ).comp( 0 ), *prior );
    f.physical->add_lin_comb( -1, *seed, *prior );
    if ( !( f.physical->norm( *prior ) < tolerance ) )
        throw std::runtime_error( "Replacement layout moved the phase anchor" );
    old.first->lift( previous.comp( 0 ), *prior );
    if ( !rephase( f, *prior, *before ) )
        throw std::runtime_error( "Previous orbit missed the replacement itinerary" );
    f.op->F( *before, prev_p, *row );
    if ( !( std::abs( f.op->report().period - prev_period ) < tolerance && f.coords->norm( *row ) < tolerance ) )
        throw std::runtime_error( "Previous orbit changed during rechart" );
    f.coords->assign_lin_comb( 1, *root, -1, *before, *row );
    T dp = p - prev_p, new_vp = 0;
    if ( !continuation::normalize_rank1_tangent( f.coords.get(), *row, dp ) ||
         !proposed->solver.tan->apply( *root, p, *row, dp, *tangent, new_vp, T( 1e-10 ), 1000 ) )
        throw std::runtime_error( "Replacement layout oriented tangent failed" );
    f.coords->assign( *root, z );
    f.coords->assign( *before, previous );
    f.coords->assign( *tangent, v );
    vp = new_vp;
    return proposed;
}
}
#endif
