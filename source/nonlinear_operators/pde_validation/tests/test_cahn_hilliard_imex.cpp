#include "cahn_hilliard_imex_test_support.h"
using namespace ch_tests;

// Constant-mobility nonlinear CH with the quartic double-well potential. Check
// the reduced split independently against a coupled (c,mu_star) Newton solve,
// then establish first-order convergence on a fixed spatial grid. No monotone
// energy assertion is used to manufacture an accuracy result.
void operator_contract( std::shared_ptr<space_type> ops, std::shared_ptr<const model_type> model )
{
    ch::convex_split_problem<space_type> problem( ops, model );
    auto                                 bundle = problem.get_imex_operator();
    check( bundle == problem.get_imex_operator(), "CH bundle was rebuilt during access" );
    auto               p = bundle->implicit_part;
    buffer<space_type> x( *ops ), v( *ops ), plus( *ops ), minus( *ops ), f( *ops ), g( *ops ), action( *ops ),
        saved( *ops ), scratch( *ops );
    for ( auto *b : { &x, &v, &plus, &minus, &f, &g, &action, &saved, &scratch } )
        b->start_use();
    initialize( *ops, *model, *x );
    initialize( *ops, *model, *v, .1, .3 );
    ops->assign_scalar( .17, p->previous_state() );
    ops->assign_scalar( .03, p->known_rhs() );
    p->set_affine_shift( .6, .8 );
    p->set_linearization_point( *x );
    auto j = p->get_jacobi_operator();
    j->apply( *v, *action );
    ops->assign( *action, *saved );
    ops->assign_mul( 1., *x, 1e-6, *v, *plus );
    ops->assign_mul( 1., *x, -1e-6, *v, *minus );
    p->apply( *plus, *f );
    p->apply( *minus, *g );
    ops->add_mul( -1., *g, 1., *f );
    ops->scale( 5e5, *f );
    check( difference( *ops, *f, *action, *scratch ) < 1e-8, "Convex CH Jacobian finite difference mismatch" );
    p->evaluate_rate( *x, *f );
    p->set_affine_shift( -.4, .2 );
    ops->assign_scalar( 12., p->known_rhs() );
    p->apply( *plus, *g );
    p->evaluate_rate( *x, *g );
    check( difference( *ops, *f, *g, *scratch ) == 0, "CH physical rate used stage coefficients/RHS" );
    j->apply( *v, *action );
    check( difference( *ops, *saved, *action, *scratch ) == 0, "CH trial residual/rate corrupted prepared Jacobian" );
    ops->scale( .25, *v );
    j->apply( *v, *action );
    ops->scale( .25, *saved );
    check( difference( *ops, *saved, *action, *scratch ) < 1e-13, "CH Jv is not linear" );
}
struct mixed_initial
{
    model_type              model;
    space_type::vector_type in, out;
    __DEVICE_TAG__ void     operator()( std::ptrdiff_t i ) const
    {
        out( 2 * i )     = in( i );
        out( 2 * i + 1 ) = model.physical_potential( in( i ), model.grid.apply( in, i, 0, 1 ) );
    }
};
struct mixed_error
{
    model_type              model;
    space_type::vector_type before, after, mixed, out;
    __DEVICE_TAG__ void     operator()( std::ptrdiff_t i ) const
    {
        const double star = model.convex_potential( after( i ), model.grid.apply( after, i, 0, 1 ) ) - before( i );
        out( i )          = std::abs( after( i ) - mixed( 2 * i ) ) + std::abs( star - mixed( 2 * i + 1 ) );
    }
};
void mixed_equivalence( std::shared_ptr<space_type> ops, std::shared_ptr<const model_type> model )
{
    buffer<space_type> before( *ops ), after( *ops ), error( *ops );
    for ( auto *b : { &before, &after, &error } )
        b->start_use();
    initialize( *ops, *model, *before );
    context<space_type> ctx( ops, model, .04 );
    const double        h         = apply( ctx, *before, *after, 0 );
    auto                mixed_ops = std::make_shared<space_type>( 2 * ops->size() );
    using P                       = mixed_reference<space_type>;
    using N                       = newton<space_type, P>;
    auto               p          = std::make_shared<P>( mixed_ops, model );
    buffer<space_type> state( *mixed_ops );
    state.start_use();
    execute( *ops, ops->size(), mixed_initial{ *model, *before, *state } );
    mixed_ops->assign( *state, p->previous_state() );
    const auto shift = ts::detail::balanced_affine_shift( 1 / h, 1. );
    p->set_affine_shift( shift.a, shift.b );
    auto solver = N::make( mixed_ops, std::make_shared<typename N::linear>( mixed_ops ) );
    solver->set_nonlinear_operator( p );
    check( solver->solve( *state ), "Independent mixed nonlinear CH correction failed" );
    execute( *ops, ops->size(), mixed_error{ *model, *before, *after, *state, *error } );
    check(
        ops->norm( *error ) / std::sqrt( double( ops->size() ) ) < 2e-10,
        "Reduced CH differs from coupled convex-split update"
    );
}
int main()
try
{
    backend_type::init_device();
    auto model = std::make_shared<const model_type>( 8 );
    auto ops   = std::make_shared<space_type>( model->grid.size() );
    operator_contract( ops, model );
    mixed_equivalence( ops, model );
    buffer<space_type> coarse( *ops ), fine( *ops ), reference( *ops ), refined( *ops ), scratch( *ops );
    for ( auto *b : { &coarse, &fine, &reference, &refined, &scratch } )
        b->start_use();
    trajectory( ops, model, .04, 5, *coarse );
    trajectory( ops, model, .02, 10, *fine );
    trajectory( ops, model, .00125, 160, *reference );
    trajectory( ops, model, .000625, 320, *refined );
    const double ec = difference( *ops, *coarse, *refined, *scratch ),
                 ef = difference( *ops, *fine, *refined, *scratch );
    check( ec / ef > 1.65 && ec / ef < 2.7, "Convex-split Euler did not converge at first order" );
    check(
        difference( *ops, *reference, *refined, *scratch ) < .12 * ef,
        "CH reference integration is insufficiently refined"
    );
    std::cout << "ORDER IMEX_EULER observed=" << std::log2( ec / ef ) << " coarse_error=" << ec << " fine_error=" << ef
              << '\n';
    finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
