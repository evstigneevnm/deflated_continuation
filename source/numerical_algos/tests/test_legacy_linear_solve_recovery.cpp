#include <iostream>
#include <scfd/utils/log_std.h>
#include <numerical_algos/linear_solve_recovery.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/iter_solver_base.h>
#include "nmfd_cpu_reference_vector_space.h"

namespace
{

using space_type   = nmfd::tests::cpu_reference_vector_space<double>;
using log_type     = scfd::utils::log_std;
using monitor_type = numerical_algos::lin_solvers::default_monitor<space_type, log_type>;

struct identity_operator
{
    void apply( const space_type::vector_type &input, space_type::vector_type &output ) const
    {
        output = input;
    }
};

struct identity_preconditioner
{
    void set_operator( const identity_operator * )
    {
    }
    void apply( space_type::vector_type & ) const
    {
    }
};

class recoverable_solver : public numerical_algos::lin_solvers::iter_solver_base<
                               identity_operator, identity_preconditioner, space_type, monitor_type, log_type>
{
    using parent_type = numerical_algos::lin_solvers::iter_solver_base<
        identity_operator, identity_preconditioner, space_type, monitor_type, log_type>;

public:
    recoverable_solver( const space_type *space, log_type *log ) : parent_type( space, log, 0, "legacy_recovery::" )
    {
    }

    bool solve( const identity_operator &, const vector_type &, vector_type & ) const override
    {
        ++solve_count;
        return this->prec_ == nullptr;
    }

    bool has_preconditioner() const
    {
        return this->prec_ != nullptr;
    }
    mutable unsigned int solve_count = 0;
};

}

int main()
{
    space_type              space( 1 );
    log_type                log;
    identity_operator       op;
    identity_preconditioner prec;
    recoverable_solver      solver( &space, &log );
    solver.set_preconditioner( &prec );
    space_type::vector_type rhs( 1, 1.0 ), solution( 1, 0.0 );
    const bool              recovered = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
        &solver, [&]() { return solver.solve( op, rhs, solution ); }, [&]() { solution.assign( 1, 0.0 ); }
    );
    if ( !recovered || solver.solve_count != 2 || !solver.has_preconditioner() ||
         solver.preconditioner_bypass_enabled() )
    {
        std::cerr << "FAILED: archived solver recovery or preconditioner restoration\n";
        return 1;
    }
    std::cout << "Archived solver recovery: PASSED\n";
    return 0;
}
