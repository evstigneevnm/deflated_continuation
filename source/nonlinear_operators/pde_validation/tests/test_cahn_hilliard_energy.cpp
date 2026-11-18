#include "cahn_hilliard_imex_test_support.h"
using namespace ch_tests;

// Forward, unforced phase separation. The discrete free energy decreases while
// mass stays constant. mu_star uses c_old, not the affine stage reference w.
// The convex inequality is an upper bound, not an equality: numerical
// dissipation is allowed. Stage defects are checked independently of energy.
template <class S>
void energy_case(
    std::shared_ptr<S> ops, std::shared_ptr<const model_type> model, double h, unsigned count, bool constant = false
)
{
    context<S>                ctx( ops, model, h );
    ch::energy_diagnostics<S> observer( ops, model );
    buffer<S>                 before( *ops ), after( *ops ), scratch( *ops );
    for ( auto *b : { &before, &after, &scratch } )
        b->start_use();
    initialize( *ops, *model, *before, .2, constant ? 0 : .12 );
    const double mass         = model->cell_area() * ops->sum( *before );
    double       first_energy = 0, last_energy = 0, time = 0;
    std::size_t  allocated = 0;
    for ( unsigned k = 0; k < count; ++k )
    {
        const double used   = apply( ctx, *before, *after, time );
        const auto   report = observer.evaluate_step( *before, *after, used );
        check( report.step_defect_norm < 2e-10, "CH energy check hid an inaccurate nonlinear solve" );
        check( std::abs( report.residual_work ) < 2e-10, "CH energy residual-work allowance is too large" );
        check( report.dissipation >= -report.roundoff_budget, "CH Laplacian has negative dissipation" );
        check(
            report.dissipation_bound_defect <= report.residual_work + report.roundoff_budget,
            "CH discrete energy inequality violated"
        );
        check(
            report.energy_after <= report.energy_before + std::abs( report.residual_work ) + report.roundoff_budget,
            "CH free energy increased"
        );
        check( std::abs( report.mass_after - mass ) < 2e-10, "CH phase separation lost mass" );
        if ( k == 0 )
            first_energy = report.energy_before;
        last_energy = report.energy_after;
        time += used;
        ctx.step->finalize( adaptation_status::accepted, time, *after );
        ops->assign( *after, *before );
        if constexpr ( std::is_same_v<typename S::backend_type, guarded_backend> )
        {
            if ( k == 0 )
                allocated = guarded_memory::allocation_count;
            check(
                allocated == guarded_memory::allocation_count, "CH stage/energy observer allocated vectors after warmup"
            );
        }
    }
    if ( !constant )
        check( last_energy < first_energy, "CH nonconstant state showed no energy decay" );
    else
        check( std::abs( last_energy - first_energy ) < 1e-12, "CH homogeneous equilibrium changed energy" );
    std::cout << "ENERGY n=" << model->grid.nx << " h=" << h << " steps=" << count << " before=" << first_energy
              << " after=" << last_energy << '\n';
}
int main()
try
{
    backend_type::init_device();
    for ( unsigned n : { 8, 12 } )
    {
        auto model = std::make_shared<const model_type>( n );
        auto ops   = std::make_shared<space_type>( model->grid.size() );
        for ( double h : { .01, .1, 1. } )
            energy_case( ops, model, h, 6 );
        energy_case( ops, model, .1, 3, true );
    }
    {
        auto model = std::make_shared<const model_type>( 64 );
        auto ops   = std::make_shared<guarded_space>( model->grid.size() );
        energy_case( ops, model, 1e-6, 3 );
    }
    finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
