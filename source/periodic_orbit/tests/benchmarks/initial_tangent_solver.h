#ifndef PERIODIC_ORBIT_BENCHMARK_INITIAL_TANGENT_SOLVER_H
#define PERIODIC_ORBIT_BENCHMARK_INITIAL_TANGENT_SOLVER_H
#include "../newton_test_support.h"

namespace orbit_benchmarks
{
// Initial tangent: solve R_z w = -R_lambda on the matching space.
template <class Space, class Orbit>
struct initial_tangent_solver
{
    using gmres_type = typename section_tests::newton_types<Space, Orbit>::linear;
    gmres_type gmres;
    unsigned successful = 0;
    explicit initial_tangent_solver( const std::shared_ptr<Space> &s )
        : gmres( s, nullptr, [&] {
              typename gmres_type::params p;
              p.basis_size = std::min<std::size_t>( 64, s->get_vector_size() );
              p.batch_size = 1;
              return p;
          }() )
    {}
    auto *get_linsolver_handle_original() { return &gmres; }
    bool solve( Orbit &op, const typename Space::vector_type &rhs, typename Space::vector_type &out )
    {
        gmres.set_operator( op.get_jacobi_operator() );
        const bool solved = gmres.solve( rhs, out );
        successful += solved;
        return solved;
    }
};
}
#endif
