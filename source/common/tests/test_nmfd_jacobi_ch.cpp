#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT
#include <cmath>
#include <memory>
#include <type_traits>
#include <scfd/backend/backend.h>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <contrib/nmfd-newton/test/solvers/newton_test_fixture.h>
#include <contrib/nmfd-newton/test/cahn_hilliard/cahn_hilliard_op.h>
#include <contrib/nmfd-newton/test/cahn_hilliard/jacobi_pre.h>

namespace
{

template <class T, class Backend>
class halo_space : public nmfd::rect_vector_space<T, 3, 2, Backend, int>
{
    using parent_t = nmfd::rect_vector_space<T, 3, 2, Backend, int>;

public:
    using parent_t::parent_t;
    using typename parent_t::vector_type;
    void init_vector( vector_type &x ) const override
    {
        const auto one = parent_t::idx_nd_type::make_ones();
        x.init( this->get_size() + 2 * one, -one );
    }
};

struct local_distributor
{
    template <class Vector>
    void sync( const Vector & ) const
    {
    }
};

template <class T, class Tensor>
struct zero_rhs
{
    __DEVICE_TAG__ Tensor operator()( T, T, T ) const
    {
        return Tensor{ T( 0 ), T( 0 ) };
    }
};

template <class Space>
struct initial_state
{
    using T = typename Space::scalar_type;
    typename Space::vector_type state;
    T                           amplitude;
    __DEVICE_TAG__ void         operator()( typename Space::idx_nd_type idx ) const
    {
        const T phi = T( .2 );
        const T mode =
            amplitude * scfd::utils::scalar_traits<T>::cos( ( T( idx[0] ) + T( .5 ) ) * T( .78539816339744830962L ) );
        scfd::static_vec::vec<T, 2> value{ ( phi * phi - T( 1 ) ) * phi + mode, phi + mode };
        state.set_vec( value, idx );
    }
};

template <class T>
void test( nmfd::tests::test_report &report )
{
    using backend_type   = scfd::backend::current;
    using space_type     = halo_space<T, backend_type>;
    using tensor_type    = scfd::static_vec::vec<T, 2>;
    using log_type       = scfd::utils::log_std;
    using potential_type = ::tests::double_well_potential<T>;
    using mobility_type  = ::tests::constant_mobility<T>;
    using rate_type      = ::tests::time_derivative<space_type, tensor_type>;
    using operator_type =
        ::tests::jacobi_op<space_type, log_type, potential_type, rate_type, mobility_type, local_distributor>;
    using problem_type = ::tests::cahn_hilliard_op<
        space_type, operator_type, log_type, potential_type, zero_rhs<T, tensor_type>, rate_type, mobility_type>;
    using prec_type =
        ::tests::jacobi_pre<space_type, log_type, potential_type, rate_type, mobility_type, local_distributor>;
    using prec_base_type = nmfd::preconditioners::preconditioner_interface<space_type, operator_type>;
    using monitor_type   = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using solver_type    = nmfd::solvers::jacobi<space_type, operator_type, prec_base_type, monitor_type, log_type>;
    using base_type =
        nmfd::solvers::iter_solver_base<space_type, monitor_type, log_type, operator_type, prec_base_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, base_type>;
    using newton_type    = nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, iteration_type>;

    auto space       = std::make_shared<space_type>( typename space_type::idx_nd_type{ 4, 4, 4 } );
    auto distributor = std::make_shared<local_distributor>();
    auto rate        = std::make_shared<rate_type>( space );
    rate->set_dt_inf( T( 100000 ) );
    int                                    neumann[3][2] = { { 1, 1 }, { 1, 1 }, { 1, 1 } };
    ::tests::boundary_cond<space_type>     boundary( neumann, neumann );
    typename operator_type::grid_step_type grid_step{ T( .25 ), T( .25 ), T( .25 ) };
    auto op      = std::make_shared<operator_type>( space, grid_step, boundary, distributor, rate );
    auto problem = std::make_shared<problem_type>( space, grid_step, boundary, op, rate );
    nmfd::detail::vector_wrap<space_type, false, false> reference( *space ), x( *space ), residual( *space );
    reference.init();
    reference.start_use();
    x.init();
    x.start_use();
    residual.init();
    residual.start_use();
    typename backend_type::template for_each_nd_type<3> for_each;
    for_each( initial_state<space_type>{ *reference, T( 0 ) }, space->get_size() );
    rate->set_previous_state( *reference );
    const T                      tolerance = std::is_same_v<T, float> ? T( 2e-4 ) : T( 1e-9 );
    typename solver_type::params p;
    p.monitor.rel_tol                    = std::is_same_v<T, float> ? T( 1e-4 ) : T( 1e-10 );
    p.monitor.abs_tol                    = T( 0 );
    p.monitor.max_iters_num              = 2000;
    p.monitor.out_min_resid_norm         = true;
    auto                       diagonal  = std::make_shared<prec_type>( op, distributor );
    std::shared_ptr<base_type> linear    = std::make_shared<solver_type>( op, space, nullptr, p, diagonal );
    auto                       iteration = std::make_shared<iteration_type>( space, linear );
    newton_type                newton( space, nullptr, iteration );
    newton.get_convergence_strategy_handle()->set_convergence_constants(
        tolerance, 30, T( 1 ), 0, T( 1 ), false, false
    );
    // Use implicit stages for which the supplied block-Jacobi smoother contracts.
    for ( T inverse_dt : { T( 100000 ), T( 200000 ) } )
    {
        rate->set_dt_inf( inverse_dt );
        for_each( initial_state<space_type>{ *x, T( .01 ) }, space->get_size() );
        problem->apply( *x, *residual );
        report.check<T>( space->norm( *residual ) > tolerance, "CH initial perturbation is nontrivial" );
        const bool solved = newton.solve( problem.get(), nullptr, nullptr, *x );
        if ( !solved )
            std::printf(
                "CH Jacobi failure: Newton status=%d, linear iterations=%d, residual=%g\n",
                newton.get_convergence_strategy_handle()->get_result_status(), linear->monitor().iters_performed(),
                double( linear->monitor().resid_norm() )
            );
        report.check<T>( solved, "CH Newton with migrated Jacobi" );
        problem->apply( *x, *residual );
        report.check<T>( space->norm( *residual ) <= T( 2 ) * tolerance, "CH true nonlinear residual" );
        space->add_lin_comb( T( -1 ), *reference, T( 1 ), *x );
        report.check<T>( space->norm( *x ) <= T( 20 ) * tolerance, "CH analytical homogeneous equilibrium" );
    }
    linear->monitor().set_temp_max_iterations( 0 );
    for_each( initial_state<space_type>{ *x, T( .01 ) }, space->get_size() );
    space->assign( *x, *residual );
    report.check<T>(
        !newton.solve( problem.get(), nullptr, nullptr, *x ), "Jacobi exhaustion propagates to CH Newton"
    );
    space->add_lin_comb( T( -1 ), *residual, T( 1 ), *x );
    report.check<T>( space->norm( *x ) == T( 0 ), "Linear failure preserves the CH Newton state" );
}

}

int main()
{
    scfd::backend::current::init_device();
    nmfd::tests::test_report report;
    test<float>( report );
    test<double>( report );
    return report.finish();
}
