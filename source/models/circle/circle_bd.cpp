#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// problem dependant
#include <common/scalar_math.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/system_operator.h>
// problem dependant ends
#include "circle_linear_solvers.h"

// problem dependant
#include <common/gpu_file_operations.h>
#include "circle_backend_typedefs.h"
#if defined( CIRCLE_VECTOR_BACKEND_HIP )
#    include <scfd/utils/init_hip.h>
#elif !defined( CIRCLE_VECTOR_BACKEND_OMP ) && !defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
#    include <scfd/utils/init_cuda.h>
#endif
// problem dependant ends

#include <main/deflation_continuation.hpp>
#include <main/nmfd_newton_types.h>
#include <main/parameters.hpp>
#include <nmfd/detail/vector_wrap.h>

namespace
{

void print_usage( const char *executable )
{
    printf(
        "Usage: %s [path_to_config_file.json] [--analytical-solution] [--seed-exact-curve lambda sign] [--quiet]\n",
        executable
    );
}

template <class T>
T parse_scalar( const std::string &value, const char *label )
{
    std::istringstream stream( value );
    T                  result = T( 0 );
    stream >> result;
    if ( !stream )
    {
        throw std::runtime_error( std::string( "failed to parse " ) + label + " from '" + value + "'" );
    }
    return result;
}

bool backend_needs_device_init()
{
#if defined( CIRCLE_VECTOR_BACKEND_OMP ) || defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
    return false;
#else
    return true;
#endif
}

int init_device_from_config( int pci_id )
{
#if defined( CIRCLE_VECTOR_BACKEND_HIP )
    return scfd::utils::init_hip( pci_id );
#elif defined( CIRCLE_VECTOR_BACKEND_OMP ) || defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
    (void)pci_id;
    return -1;
#else
    return scfd::utils::init_cuda( pci_id );
#endif
}

} // namespace

int main( int argc, char const *argv[] )
{
    size_t Nx = 1; // size of the vector variable. 1 in this case

    typedef gpu_file_operations<vec_ops_real> files_ops_t;
    using monitor_t = nmfd::solvers::default_monitor<vec_ops_real, log_t>;

    typedef nonlinear_operators::circle<vec_ops_real, Blocks_x_> circle_t;

    typedef nonlinear_operators::linear_operator_circle<vec_ops_real, circle_t> lin_op_t;

    typedef nonlinear_operators::preconditioner_circle<vec_ops_real, circle_t, lin_op_t> prec_t;

#if defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
    typedef double                           parameters_real;
    typedef main_classes::parameters<double> parameters_t;
#else
    typedef real                           parameters_real;
    typedef main_classes::parameters<real> parameters_t;
#endif
    std::string path_to_config_file     = "json_project_files/circle_test.json";
    bool        use_analytical_solution = false;
    bool        seed_exact_curve        = false;
    bool        quiet                   = false;
    real        seed_lambda             = real( 0 );
    real        seed_sign               = real( 1 );
    for ( int argi = 1; argi < argc; ++argi )
    {
        const std::string arg = argv[argi];
        if ( arg == "--analytical-solution" )
        {
            use_analytical_solution = true;
        }
        else if ( arg == "--quiet" )
        {
            quiet = true;
        }
        else if ( arg == "--seed-exact-curve" )
        {
            if ( argi + 2 >= argc )
            {
                print_usage( argv[0] );
                return 1;
            }
            try
            {
                seed_lambda = parse_scalar<real>( argv[++argi], "seed lambda" );
                seed_sign   = parse_scalar<real>( argv[++argi], "seed sign" );
            }
            catch ( const std::exception &e )
            {
                std::cerr << e.what() << std::endl;
                return 2;
            }
            seed_exact_curve = true;
        }
        else if ( path_to_config_file == "json_project_files/circle_test.json" )
        {
            path_to_config_file = arg;
        }
        else
        {
            print_usage( argv[0] );
            return 1;
        }
    }
    std::cout << "Using circle backend: " << CIRCLE_BACKEND_NAME << std::endl;
    std::cout << "Analytical seed: " << ( use_analytical_solution ? "enabled" : "disabled" ) << std::endl;
    if ( seed_exact_curve )
    {
        std::cout << "Exact regular-continuation seed: lambda=" << seed_lambda << ", sign=" << seed_sign << std::endl;
    }
    std::cout << "Reading config file: " << path_to_config_file << std::endl;
    auto parameters =
        std::make_shared<parameters_t>( main_classes::read_parameters_json<parameters_real>( path_to_config_file ) );
    if ( !quiet )
    {
        parameters->plot_all();
    }

    Nx = parameters->nonlinear_operator.N_size.at( 0 ) == 1 ? Nx
                                                            : ( throw std::runtime_error(
                                                                  "incorrect size for problem in config "
                                                                  "file provided. Expecting 1."
                                                              ) );

    unsigned int m_Krylov                     = parameters->stability_continuation.Krylov_subspace;
    int          nvidia_pci_id                = parameters->nvidia_pci_id;
    bool         use_high_precision_reduction = parameters->use_high_precision_reduction;

    if ( backend_needs_device_init() )
    {
        const int device = init_device_from_config( nvidia_pci_id );
        std::cout << "Using device " << device << std::endl;
    }
    real norm_wight = common::scalar_math::sqrt( real( Nx ) );
    (void)norm_wight;
    real Rad = 1.0;

    auto vec_ops_R = std::make_shared<vec_ops_real>( Nx );
    if ( use_high_precision_reduction )
    {
#if defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
        std::cerr << "Warning: variable-precision vector operations do not expose high-precision reduction toggles; "
                     "using their native reductions.\n";
#else
        vec_ops_R->use_high_precision();
#endif
    }
    auto file_ops = std::make_shared<files_ops_t>( vec_ops_R.get() );

    auto  CIRCLE = std::make_shared<circle_t>( Rad, Nx, vec_ops_R );
    log_t log;
    log_t log_linsolver;
    log.set_verbosity( quiet ? 0 : 1 );
    log_linsolver.set_verbosity( quiet ? 0 : 1 );

    // test continuaiton process of a single curve
    using linear_system_type =
        nmfd::solvers::sherman_morrison<vec_ops_real, monitor_t, log_t, lin_op_t, prec_t, nmfd::solvers::cgs>;
    using storage_type = deflation::solution_storage<vec_ops_real, log_t>;
#if defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
    // NMFD's default convergence strategy still needs a separate multiprecision-math pass.
    using newton_types = void;
#else
    using newton_types =
        main_classes::nmfd_newton_types<vec_ops_real, log_t, circle_t, lin_op_t, linear_system_type, storage_type>;
#endif
    using deflation_continuation_t = main_classes::deflation_continuation<
        vec_ops_real, files_ops_t, log_t, monitor_t, circle_t, lin_op_t, prec_t, circle_model::cgs,
        nonlinear_operators::system_operator, parameters_t, storage_type, continuation::system_operator_continuation,
        linear_system_type, newton_types>;

    std::cout << "Newton solver: " << ( deflation_continuation_t::uses_nmfd_newton ? "NMFD" : "project legacy" )
              << '\n';
    deflation_continuation_t DC( vec_ops_R, file_ops, &log, &log_linsolver, CIRCLE, parameters );

    DC.set_parameters();
    DC.use_analytical_solution( use_analytical_solution );

    if ( seed_exact_curve )
    {
        const real radicand = Rad * Rad - seed_lambda * seed_lambda;
        if ( radicand < real( 0 ) )
        {
            std::cerr << "Exact seed lambda is outside the circle: " << seed_lambda << std::endl;
            return 2;
        }
        nmfd::detail::vector_wrap<vec_ops_real, false, false> x_seed( *vec_ops_R );
        x_seed.init();
        x_seed.start_use();
        vec_ops_R->assign_scalar( seed_sign * common::scalar_math::sqrt( radicand ), *x_seed );
        DC.add_solution_curve( *x_seed, seed_lambda );
    }

    DC.execute();

    return 0;
}
