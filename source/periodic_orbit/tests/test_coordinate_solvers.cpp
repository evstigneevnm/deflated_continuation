#include "section_test_support.h"
#include <nmfd/operations/glued_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <scfd/utils/log_std.h>
#include <periodic_orbit/sections/affine_chart.h>
using namespace section_tests;

using blocks       = nmfd::operations::glued_vector_space<operations>;
using block_vector = blocks::vector_type;
using block_wrap   = nmfd::detail::vector_wrap<blocks, false, false>;

// Manufactured cyclic coordinate equations R_i=2*(z_i-z_i*)-.25*(z_{i-1}-z_{i-1}*).
// They exercise runtime section storage through unmodified NMFD GMRES/Newton.
// This is an algebra/ownership gate, not a periodic-orbit discovery test.
struct cyclic_problem
{
    struct jacobian
    {
        std::shared_ptr<blocks> space;
        void                    apply( const block_vector &in, block_vector &out ) const
        {
            const auto count = space->component_count();
            for ( std::size_t i = 0; i < count; ++i )
                space->space_comp( i ).assign_lin_comb(
                    2, in.comp( i ), -.25, in.comp( i ? i - 1 : count - 1 ), out.comp( i )
                );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<blocks>   space;
    block_wrap                exact, scratch;
    std::shared_ptr<jacobian> jacobi;
    explicit cyclic_problem( std::shared_ptr<blocks> s )
        : space( std::move( s ) ), exact( *space ), scratch( *space ),
          jacobi( std::make_shared<jacobian>( jacobian{ space } ) )
    {
        exact.init();
        exact.start_use();
        scratch.init();
        scratch.start_use();
        for ( std::size_t i = 0; i < space->component_count(); ++i )
            space->space_comp( i ).assign_scalar( i + 1, ( *exact ).comp( i ) );
    }
    void apply( const block_vector &in, block_vector &residual )
    {
        space->assign_lin_comb( 1, in, -1, *exact, *scratch );
        jacobi->apply( *scratch, residual );
    }
    void set_linearization_point( const block_vector & )
    {
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
};

void test()
{
    using log         = scfd::utils::log_std;
    using monitor     = nmfd::solvers::monitor_krylov<blocks, log>;
    using linear      = nmfd::solvers::gmres<blocks, monitor, log, cyclic_problem::jacobian>;
    using iteration   = nmfd::solvers::newton_iteration<blocks, cyclic_problem, linear>;
    using convergence = nmfd::solvers::default_convergence_strategy<blocks, log, cyclic_problem>;
    using nonlinear   = nmfd::solvers::nonlinear_solver<blocks, log, cyclic_problem, iteration>;
    auto physical = std::make_shared<operations>( 2 ), coordinates = std::make_shared<operations>( 1 );
    for ( std::size_t count : { 1u, 2u, 4u, 8u } )
    {
        auto           space   = std::make_shared<blocks>( coordinates, count );
        auto           problem = std::make_shared<cyclic_problem>( space );
        linear::params lp;
        lp.basis_size            = 8;
        lp.batch_size            = 1;
        lp.monitor.abs_tol       = 1e-13;
        lp.monitor.rel_tol       = 1e-12;
        lp.monitor.max_iters_num = 100;
        auto                lin  = std::make_shared<linear>( space, nullptr, lp );
        auto                iter = std::make_shared<iteration>( space, lin );
        convergence::params cp;
        cp.abs_tol       = 1e-10;
        cp.rel_tol       = 1e-12;
        cp.max_iters_num = 4;
        nonlinear newton( space, nullptr, iter, std::make_shared<convergence>( space, nullptr, cp ) );
        newton.set_nonlinear_operator( problem );
        block_wrap initial( *space ), answer( *space ), residual( *space );
        initial.init();
        answer.init();
        residual.init();
        initial.start_use();
        answer.start_use();
        residual.start_use();
        space->assign_scalar( 0, *initial );
        require( newton.solve( *initial, *answer ), "Newton does not support runtime section coordinates" );
        problem->apply( *answer, *residual );
        require( space->norm( *residual ) < 1e-9, "Runtime coordinate true residual" );
        space->assign_lin_comb( 1, *answer, -1, *problem->exact, *residual );
        require( space->norm( *residual ) < 1e-9, "Runtime coordinate solution error" );
        require( newton.solve( *initial, *answer ), "Repeated runtime-coordinate solve failed" );
        auto state = buffer( physical ), normal = buffer( physical ), center = buffer( physical );
        fill( *physical, *center, 1, 0 );
        fill( *physical, *normal, 0, 1 );
        auto                                     section = std::make_shared<plane>( physical, *center, *normal );
        periodic_orbit::affine_chart<operations> chart( physical, coordinates, section );
        plane::evaluation_workspace              work( physical );
        for ( std::size_t i = 0; i < count; ++i )
        {
            chart.lift( ( *answer ).comp( i ), *state );
            require(
                std::abs( section->bind( work ).value( 0, *state ) ) < 1e-12, "Solved coordinate left physical section"
            );
        }
    }
    require( guarded_memory::host_bytes == 0, "Coordinate solver used host vector transfers" );
}
int main()
{
    return run( test );
}
