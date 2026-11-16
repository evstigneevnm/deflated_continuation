#include <nonlinear_operators/pde_validation/cahn_hilliard.h>
#include "pde_test_support.h"
using namespace pde_tests;

struct algebraic_difference
{
    space_type::vector_type x, y, out;
    __DEVICE_TAG__ void     operator()( std::ptrdiff_t i ) const
    {
        out( 2 * i )     = 0;
        out( 2 * i + 1 ) = x( 2 * i + 1 ) - y( 2 * i + 1 );
    }
};

void unscaled_rows(
    std::shared_ptr<space_type> ops, std::shared_ptr<pde::cahn_hilliard<space_type>> p,
    const pde::cahn_hilliard_model<double> &model
)
{
    buffer<space_type> x( *ops ), first( *ops ), second( *ops ), scratch( *ops );
    for ( auto *b : { &x, &first, &second, &scratch } )
        b->start_use();
    exact( *ops, model, *x, .3 );
    ops->assign_scalar( 0., p->previous_state() );
    ops->assign_scalar( 0., p->known_rhs() );
    p->set_affine_shift( .4, .8 );
    p->apply( *x, *first );
    // Poison the algebraic reference/RHS too: these slots must not be used to
    // formulate or rescale a constraint. Differential rows may change freely.
    ops->assign_scalar( 1., p->previous_state() );
    ops->assign_scalar( 2., p->known_rhs() );
    p->set_affine_shift( -.2, .1 );
    p->apply( *x, *second );
    execute( *ops, model.grid.size(), algebraic_difference{ *first, *second, *scratch } );
    check( ops->norm( *scratch ) == 0, "CH algebraic row depends on shift/reference/RHS" );
}

// Linear mixed index-1 Cahn-Hilliard on a periodic grid. The initial chemical
// potential is consistent; both modes decay with the discrete dispersion law.
// Verify mean concentration and the unscaled algebraic equation at every commit.
template <engine E, bool Adaptive, class S>
report
run( std::shared_ptr<S> ops, std::shared_ptr<const pde::cahn_hilliard_model<double>> model, const char *method,
     double h, double tol = 1e-6, double end = .62 )
{
    return integrate<E, Adaptive, false>(
        ops, model, std::make_shared<pde::cahn_hilliard<S>>( ops, model ), method, h, tol, end
    );
}
template <engine E>
void accuracy(
    std::shared_ptr<space_type> ops, std::shared_ptr<const pde::cahn_hilliard_model<double>> model, const char *method,
    int order
)
{
    const auto a = run<E, false>( ops, model, method, .08 ), b = run<E, false>( ops, model, method, .04 );
    std::cout << "ORDER " << method << " observed=" << std::log2( a.error / b.error ) << " expected=" << order << '\n';
    check( a.error / b.error > std::pow( 2., order - .45 ), "Mixed CH temporal order regression" );
    run<E, false>( ops, model, method, -.04, 1e-6, .3 );
}
int main()
try
{
    backend_type::init_device();
    auto model = std::make_shared<const pde::cahn_hilliard_model<double>>( 8 );
    auto ops   = std::make_shared<space_type>( 2 * model->grid.size() );
    auto p     = std::make_shared<pde::cahn_hilliard<space_type>>( ops, model );
    contract( ops, p, *model );
    unscaled_rows( ops, p, *model );
    lifecycle( ops, p, *model );
    using C    = ts::integration::time_step_adaptation_constant<space_type>;
    using N    = newton<space_type, pde::cahn_hilliard<space_type>>;
    using Step = ts::runge_kutta::implicit_time_step<space_type, pde::cahn_hilliard<space_type>, C, typename N::solver>;
    bool refused = false;
    try
    {
        auto c = std::make_shared<C>();
        auto l = std::make_shared<typename N::linear>( ops );
        Step unsupported( ops, p, c, N::make( ops, l ), typename Step::params{ "ESDIRK2(1)3" } );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused, "Mixed CH accepted a method requiring an absent consistent rate" );
    accuracy<engine::dirk>( ops, model, "SDIRK2(1)2", 2 );
    accuracy<engine::bdf>( ops, model, "BDF2", 2 );
    accuracy<engine::rosenbrock>( ops, model, "ROS34PW2", 3 );
    const auto a = run<engine::bdf, true>( ops, model, "BDF", .08, 1e-3 );
    const auto b = run<engine::bdf, true>( ops, model, "BDF", .08, 1e-7 );
    check( b.error < a.error * .6, "Mixed CH adaptive BDF refinement failed" );
    auto large  = std::make_shared<const pde::cahn_hilliard_model<double>>( 64 );
    auto device = std::make_shared<guarded_space>( 2 * large->grid.size() );
    run<engine::dirk, false>( device, large, "SDIRK2(1)2", 1e-6, 1e-6, .300002 );
    run<engine::bdf, false>( device, large, "BDF2", 1e-6, 1e-6, .300002 );
    run<engine::rosenbrock, false>( device, large, "ROS34PW2", 1e-6, 1e-6, .300002 );
    device.reset();
    finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
