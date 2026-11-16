#include <nonlinear_operators/pde_validation/diffusion.h>
#include "pde_test_support.h"

// Forced heat equation with two independently decaying discrete Fourier modes.
// Exact semidiscrete forcing separates temporal error from spatial error.
int main()
try
{
    backend_type::init_device();
    bool rejected = false;
    try
    {
        discretization::finite_difference::periodic_laplacian_2d<double> invalid( 0, 8, 1., 1. );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    pde_tests::check( rejected, "Invalid periodic grid accepted" );
    pde_tests::ode_suite<nonlinear_operators::pde_validation::diffusion<double>>();
    pde_tests::finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
