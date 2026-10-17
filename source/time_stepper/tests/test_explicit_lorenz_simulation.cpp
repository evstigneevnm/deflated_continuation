#include <memory>
#include "explicit_lorenz_problem.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>
#include <scfd/static_vec/vec.h>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type                 = scfd::backend::cuda<std::ptrdiff_t>;
constexpr const char *backend_name = "cuda";
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type                 = scfd::backend::omp<std::ptrdiff_t>;
constexpr const char *backend_name = "omp";
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type                 = scfd::backend::serial_cpu<std::ptrdiff_t>;
constexpr const char *backend_name = "serial_cpu";
#endif
#include <common/scfd_vector_operations.h>
#include <contrib/json/nlohmann/json.hpp>
#include <time_stepper/integration/time_integrator.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

using operations_type = scfd_vector_operations<backend_type, double>;

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template <class VectorOperations>
class external_manager
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string filename;
    };

    external_manager( std::shared_ptr<VectorOperations> operations, const params &p, const nlohmann::json &metadata )
        : operations_( std::move( operations ) ), params_( p ), metadata_( metadata )
    {
        trajectory_.reserve( 1024 );
    }

    void set_time_interval( const scalar_type &start, scalar_type &end )
    {
        start_ = start;
        end_   = end;
    }

    bool apply( const vector_type &in, vector_type &out )
    {
        if ( trajectory_.empty() )
        {
            record( start_, in );
        }
        record( end_, out );
        return false;
    }

    integration_status get_status() const
    {
        return integration_status::running;
    }

    void complete( std::size_t accepted_steps )
    {
        if ( trajectory_.size() != accepted_steps + 1 )
        {
            throw std::runtime_error( "Trajectory row count differs from accepted steps" );
        }
        std::ofstream output;
        output.exceptions( std::ios::failbit | std::ios::badbit );
        output.imbue( std::locale::classic() );
        output.open( params_.filename );
        output << std::setprecision( std::numeric_limits<scalar_type>::max_digits10 );
        output << "# " << metadata_.dump() << '\n';
        output << "# time x y z\n";
        for ( const auto &sample : trajectory_ )
        {
            output << sample.time << ' ' << sample.state[0] << ' ' << sample.state[1] << ' ' << sample.state[2] << '\n';
        }
        const nlohmann::json footer = { { "status", "completed" }, { "rows", trajectory_.size() } };
        output << "# " << footer.dump() << '\n';
        output.close();
    }

private:
    struct sample
    {
        scalar_type                           time;
        scfd::static_vec::vec<scalar_type, 3> state;
    };

    std::shared_ptr<VectorOperations> operations_;
    params                            params_;
    nlohmann::json                    metadata_;
    std::vector<sample>               trajectory_;
    scalar_type                       start_ = 0, end_ = 0;

    void record( scalar_type time, const vector_type &state )
    {
        const auto view = operations_->view( state );
        trajectory_.push_back( { time, { view( 0 ), view( 1 ), view( 2 ) } } );
    }
};
}
}
}

template <class Adaptation>
void run_simulation(
    std::shared_ptr<operations_type> operations_owner, double end_time, const std::string &filename,
    const std::string &method, const std::string &controller, std::shared_ptr<Adaptation> adaptation_owner,
    const nlohmann::json &adaptation_settings
)
{
    auto &operations      = *operations_owner;
    auto &adaptation      = *adaptation_owner;
    using problem_type    = nmfd::time_steppers::tests::explicit_lorenz_problem<operations_type>;
    using adaptation_type = Adaptation;
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, problem_type, adaptation_type>;
    using external_type = nmfd::time_steppers::tests::external_manager<operations_type>;
    using integrator_type =
        nmfd::time_steppers::integration::time_integrator<operations_type, step_type, external_type>;

    auto                                       problem_owner = std::make_shared<problem_type>();
    auto                                      &problem       = *problem_owner;
    nmfd::detail::vector_wrap<operations_type> initial( operations ), result( operations );
    initial.start_use();
    result.start_use();
    const scfd::static_vec::vec<double, 3> initial_values{ 2.2, 30.5, 2.5 };
    operations.set( initial_values.d, *initial );

    auto step_owner = std::make_shared<step_type>(
        operations_owner, problem_owner, adaptation_owner, typename step_type::params{ method }
    );
    auto                &step     = *step_owner;
    const nlohmann::json metadata = {
        { "format", "nmfd.lorenz_trajectory.v1" },
        { "problem", "lorenz" },
        { "method", method },
        { "backend", backend_name },
        { "controller", controller },
        { "start_time", 0 },
        { "end_time", end_time },
        { "initial_state", { initial_values[0], initial_values[1], initial_values[2] } },
        { "parameters",
          { { "sigma", problem.sigma },
            { "rho", problem.rho },
            { "beta", problem.beta },
            { "epsilon", problem.epsilon },
            { "delta", problem.delta } } },
        { "adaptation", adaptation_settings }
    };
    auto external_owner =
        std::make_shared<external_type>( operations_owner, typename external_type::params{ filename }, metadata );
    auto           &external = *external_owner;
    integrator_type integrator( operations_owner, step_owner, {}, external_owner );
    integrator.set_time_interval( 0, end_time );
    integrator.apply( *initial, *result );
    if ( integrator.get_status() != nmfd::time_steppers::integration_status::completed ||
         integrator.get_final_time() != end_time )
    {
        throw std::runtime_error( "Lorenz simulation failed; trajectory is incomplete" );
    }
    external.complete( integrator.get_steps() );
    std::cout << "lorenz: " << method << ", " << controller << ", " << backend_name << ", " << integrator.get_steps()
              << " accepted steps, final time " << end_time << ", trajectory " << filename << '\n';
}

template <class Adaptation>
void run_adaptive_simulation(
    std::shared_ptr<operations_type> operations_owner, double end_time, const std::string &filename,
    const std::string &method, const std::string &controller, double initial_step
)
{
    auto                       &operations = *operations_owner;
    typename Adaptation::params p;
    p.relative_tolerance = 1e-10;
    p.absolute_tolerance = 1e-12;
    if ( initial_step > 0 )
    {
        p.initial_step = initial_step;
    }
    auto  adaptation_owner = std::make_shared<Adaptation>( operations_owner, p );
    auto &adaptation       = *adaptation_owner;
    run_simulation(
        operations_owner, end_time, filename, method, controller, adaptation_owner,
        { { "relative_tolerance", p.relative_tolerance },
          { "absolute_tolerance", p.absolute_tolerance },
          { "initial_step", p.initial_step },
          { "minimum_step", p.minimum_step },
          { "maximum_step", p.maximum_step } }
    );
}

double positive_time( const std::string &value )
{
    std::size_t  parsed = 0;
    const double time   = std::stod( value, &parsed );
    if ( parsed != value.size() || !std::isfinite( time ) || time <= 0 )
    {
        throw std::invalid_argument( "Integration time and step size must be positive and finite" );
    }
    return time;
}

int main( int argc, char **argv )
try
{
    if ( argc < 3 || argc > 6 )
    {
        std::cerr << "Usage: " << argv[0]
                  << " end_time trajectory.dat [RK45|RK23|DOP853|RK4] [matlab|scipy|constant] [step_size]\n";
        return argc == 2 && std::string( argv[1] ) == "--help" ? 0 : 2;
    }
    const std::string method     = argc >= 4 ? argv[3] : "RK45";
    const std::string controller = argc >= 5            ? argv[4]
                                   : method == "RK4"    ? "constant"
                                   : method == "DOP853" ? "scipy"
                                                        : "matlab";
    if ( ( method != "RK45" && method != "RK23" && method != "DOP853" && method != "RK4" ) ||
         ( controller != "matlab" && controller != "scipy" && controller != "constant" ) )
    {
        throw std::invalid_argument( "Expected RK45, RK23, DOP853 or RK4 and matlab, scipy or constant adaptation" );
    }
    if ( ( method == "RK4" ) != ( controller == "constant" ) )
    {
        throw std::invalid_argument( "This simulation uses constant stepping only for RK4" );
    }
    if ( method == "DOP853" && controller != "scipy" )
    {
        throw std::invalid_argument( "Adaptive DOP853 requires scipy adaptation for its combined estimator" );
    }
    const double end_time  = positive_time( argv[1] );
    const double step_size = argc == 6 ? positive_time( argv[5] ) : 0;
    backend_type::init_device();
    auto  operations_owner = std::make_shared<operations_type>( 3 );
    auto &operations       = *operations_owner;
    if ( controller == "constant" )
    {
        using adaptation_type   = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
        const double fixed_step = step_size > 0 ? step_size : .001;
        auto  adaptation_owner  = std::make_shared<adaptation_type>( typename adaptation_type::params{ fixed_step } );
        auto &adaptation        = *adaptation_owner;
        run_simulation(
            operations_owner, end_time, argv[2], method, controller, adaptation_owner, { { "step_size", fixed_step } }
        );
    }
    else if ( controller == "scipy" )
    {
        run_adaptive_simulation<nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>>(
            operations_owner, end_time, argv[2], method, controller, step_size
        );
    }
    else
    {
        run_adaptive_simulation<nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>>(
            operations_owner, end_time, argv[2], method, controller, step_size
        );
    }
}
catch ( const std::exception &e )
{
    std::cerr << "Lorenz simulation: " << e.what() << '\n';
    return 1;
}
