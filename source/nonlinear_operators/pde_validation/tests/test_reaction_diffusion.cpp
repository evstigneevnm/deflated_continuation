#include <nonlinear_operators/pde_validation/reaction_diffusion.h>
#include "pde_test_support.h"

// Nonlinear two-field Brusselator with spatially varying manufactured forcing.
// Coupled reactions are explicit in IMEX; diffusion alone is implicit. Other
// engines solve the unchanged unsplit equation and use its full Jacobian.
int main()
try
{
    backend_type::init_device();
    pde_tests::ode_suite<nonlinear_operators::pde_validation::reaction_diffusion<double>>();
    pde_tests::finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
