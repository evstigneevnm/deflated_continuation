#include <cmath>
#include <memory>
#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/log.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/solvers/bicgstab.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/cgs.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <contrib/nmfd-linsolvers/test/common/test_harness.h>

namespace
{

struct space_type : nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<std::ptrdiff_t>, double>
{
    using parent_t = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<std::ptrdiff_t>, double>;
    using parent_t::parent_t;
    std::size_t get_vector_size() const
    {
        return this->size();
    }
};

struct operator_type
{
    using vector_type = typename space_type::vector_type;
    void apply( const vector_type &in, vector_type &out ) const
    {
        for ( int i = 0; i < 8; ++i )
            out( i ) = ( 4.0 + i ) * in(i)-in( ( i + 7 ) % 8 ) + 0.5 * in( ( i + 1 ) % 8 );
    }
};

struct legacy_prec
{
    void set_operator( const operator_type * )
    {
    }
    void apply( typename space_type::vector_type & ) const
    {
    }
};

template <
    template <class, class, class, class, class> class Original,
    template <class, class, class, class, class> class Migrated, bool Polynomial = false>
void compare_solver( nmfd::tests::test_harness &test )
{
    using log_type    = scfd::utils::log_std;
    using old_monitor = numerical_algos::lin_solvers::default_monitor<space_type, log_type>;
    using new_monitor = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using old_solver  = Original<operator_type, legacy_prec, space_type, old_monitor, log_type>;
    using new_solver  = Migrated<
         space_type, new_monitor, log_type, operator_type, nmfd::preconditioners::dummy<space_type, operator_type>>;
    auto     space = std::make_shared<space_type>( 8 );
    auto     op    = std::make_shared<operator_type>();
    log_type log;
    log.set_verbosity( -1 );
    old_solver original( space.get(), &log );
    original.monitor().init( 1e-11, 1e-14, 200 );
    typename new_solver::params p;
    p.monitor.rel_tol       = 1e-11;
    p.monitor.abs_tol       = 1e-14;
    p.monitor.max_iters_num = 200;
    new_solver                                          migrated( op, space, &log, p );
    nmfd::detail::vector_wrap<space_type, false, false> exact( *space ), rhs( *space ), old_x( *space ),
        new_x( *space ), residual( *space );
    exact.init();
    exact.start_use();
    rhs.init();
    rhs.start_use();
    old_x.init();
    old_x.start_use();
    new_x.init();
    new_x.start_use();
    residual.init();
    residual.start_use();
    for ( int i = 0; i < 8; ++i )
        ( *exact )( i ) = std::sin( double( i + 1 ) );
    op->apply( *exact, *rhs );
    for ( int degree : { 1, 2, 4 } )
    {
        if constexpr ( Polynomial )
        {
            original.set_basis_size( degree );
            migrated.set_basis_size( degree );
        }
        space->assign_scalar( 0.1, *old_x );
        space->assign_scalar( 0.1, *new_x );
        test.require( original.solve( *op, *rhs, *old_x ), "Production solver converges on reference system" );
        test.require( migrated.solve( *rhs, *new_x ), "Migrated solver converges on reference system" );
        for ( auto *solution : { &*old_x, &*new_x } )
        {
            op->apply( *solution, *residual );
            space->add_lin_comb( 1.0, *rhs, -1.0, *residual );
            test.require(
                space->norm( *residual ) <= 2e-10 * space->norm( *rhs ), "Original and migrated physical residuals"
            );
        }
        space->assign_lin_comb( 1.0, *old_x, -1.0, *new_x, *residual );
        test.require(
            space->norm( *residual ) <= 1e-9 * space->norm( *exact ), "Solutions agree with archived implementation"
        );
        if constexpr ( !Polynomial )
            break;
    }
}

}

int main()
{
    nmfd::tests::test_harness test;
    compare_solver<numerical_algos::lin_solvers::bicgstab, nmfd::solvers::bicgstab>( test );
    compare_solver<numerical_algos::lin_solvers::bicgstabl, nmfd::solvers::bicgstabl, true>( test );
    compare_solver<numerical_algos::lin_solvers::cgs, nmfd::solvers::cgs>( test );
    return test.finish( "Production short-recurrence migration parity" );
}
