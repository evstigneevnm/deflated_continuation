#include "cahn_hilliard_imex_test_support.h"
#include <nonlinear_operators/cahn_hilliard/unsplit_problem.h>
#include <nonlinear_operators/cahn_hilliard/physical_diagnostics.h>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace ch_tests;
using json   = nlohmann::json;
using space  = guarded_space;
using vector = space::vector_type;

#ifdef CH_FOURIER
#    include "../fourier/tests/spatial_checks.h"
#    include "../fourier/biharmonic_preconditioner.h"
using split_problem           = ch::fourier::split_problem<space, ch_fft_backend>;
using full_problem            = ch::fourier::problem<space, ch_fft_backend, false>;
using physical_observer       = ch::fourier::diagnostics<space, ch_fft_backend>;
using energy_observer         = physical_observer;
constexpr auto spatial_method = "fourier";
template <class P>
using benchmark_newton = newton<space, P, linear_solver<space, P, ch::fourier::biharmonic_preconditioner<P>>>;
template <class P>
auto make_benchmark_linear( std::shared_ptr<space> ops )
{
    return std::make_shared<typename benchmark_newton<P>::linear>(
        ops, std::make_shared<ch::fourier::biharmonic_preconditioner<P>>()
    );
}
#else
using split_problem           = ch::convex_split_problem<space>;
using full_problem            = ch::unsplit_problem<space>;
using physical_observer       = ch::physical_diagnostics<space>;
using energy_observer         = ch::energy_diagnostics<space>;
constexpr auto spatial_method = "finite_difference";
template <class P>
using benchmark_newton = newton<space, P>;
template <class P>
auto make_benchmark_linear( std::shared_ptr<space> ops )
{
    return std::make_shared<typename benchmark_newton<P>::linear>( ops );
}
#endif

// Unforced periodic CH: c_t=m*Delta(c^3-c-epsilon^2*Delta(c)).
// All methods use the same spatial equation and initial data. IMEX_EULER alone
// has convex-split energy acceptance. Accuracy uses a common unsplit reference.
template <engine E, bool Adaptive = false>
struct benchmark_context
{
    using problem_type  = std::conditional_t<E == engine::imex, split_problem, full_problem>;
    using implicit_type = std::conditional_t<E == engine::imex, split_problem::implicit_operator_type, full_problem>;
    using newton_type   = benchmark_newton<implicit_type>;
    using control_type  = std::conditional_t<
         Adaptive, ts::integration::time_step_adaptation_scipy<space>,
         ts::integration::time_step_adaptation_constant<space>>;
    using solver_type =
        std::conditional_t<E == engine::rosenbrock, typename newton_type::linear, typename newton_type::solver>;
    using step_type = std::conditional_t<
        E == engine::imex, ts::runge_kutta::imex_time_step<space, problem_type, control_type, solver_type, Adaptive>,
        std::conditional_t<
            E == engine::dirk,
            ts::runge_kutta::implicit_time_step<space, problem_type, control_type, solver_type, Adaptive>,
            std::conditional_t<
                E == engine::bdf, ts::bdf::implicit_time_step<space, problem_type, control_type, solver_type, Adaptive>,
                ts::rosenbrock::time_step<space, problem_type, control_type, solver_type, Adaptive>>>>;
    std::shared_ptr<problem_type>                 problem;
    std::shared_ptr<control_type>                 control;
    std::shared_ptr<typename newton_type::linear> linear;
    std::unique_ptr<step_type>                    step;
    static std::shared_ptr<solver_type>
    make_solver( std::shared_ptr<space> ops, std::shared_ptr<typename newton_type::linear> linear )
    {
        if constexpr ( E == engine::rosenbrock )
            return linear;
        else
            return newton_type::make( ops, linear );
    }
    static std::shared_ptr<control_type>
    make_control( std::shared_ptr<space> ops, double h, const json &config, double scale )
    {
        typename control_type::params p;
        if constexpr ( Adaptive )
        {
            p.initial_step       = std::min( h, config.at( "initial_step" ).get<double>() );
            p.maximum_step       = h;
            p.relative_tolerance = config.at( "relative_tolerance" ).get<double>() * scale;
            p.absolute_tolerance = config.at( "absolute_tolerance" ).get<double>() * scale;
            return std::make_shared<control_type>( ops, p );
        }
        else
        {
            p.step_size = h;
            return std::make_shared<control_type>( p );
        }
    }
    static typename step_type::params parameters( const std::string &method, const json &config )
    {
        typename step_type::params p{ method };
        if constexpr ( E == engine::bdf )
        {
            p.adaptive      = Adaptive;
            p.maximum_order = config.value( "maximum_order", 5u );
        }
        return p;
    }
    benchmark_context(
        std::shared_ptr<space> ops, std::shared_ptr<const model_type> model, double h, const std::string &method,
        const json &config, double scale
    )
        : problem( std::make_shared<problem_type>( ops, model ) ), control( make_control( ops, h, config, scale ) ),
          linear( make_benchmark_linear<implicit_type>( ops ) ),
          step(
              std::make_unique<step_type>(
                  ops, problem, control, make_solver( ops, linear ), parameters( method, config )
              )
          )
    {
    }
};

unsigned method_order( const std::string &method )
{
    if ( method == "IMEX_EULER" || method == "IE" )
        return 1;
    if ( method == "CN" || method == "SDIRK2(1)2" || method == "BDF2" )
        return 2;
    if ( method == "ROS3" || method == "ROS34PW2" || method == "ARK324L2SA" || method == "IMEX_ARK324L2SA" )
        return 3;
    if ( method == "BDF" )
        return 0; // Variable accepted order, not a fixed order-five claim.
    throw std::invalid_argument( "Unsupported CH comparison method" );
}

class experiment
{
public:
    explicit experiment( const json &config )
        : config_( config ),
          model_(
              std::make_shared<const model_type>(
                  config.at( "grid" ).get<unsigned>(), config.at( "mobility" ), config.at( "epsilon" )
              )
          ),
          ops_( std::make_shared<space>( model_->grid.size() ) )
    {
    }

    template <engine E, bool Adaptive = false>
    json integrate(
        unsigned count, double end, vector &result, const std::string &method, bool record = false, double scale = 1
    )
    {
        const double                   h = end / count;
        benchmark_context<E, Adaptive> ctx( ops_, model_, h, method, config_, scale );
        buffer<space>                  state( *ops_ ), out( *ops_ );
        state.start_use();
        out.start_use();
        initialize( *ops_, *model_, *state, config_.at( "mean" ), config_.at( "amplitude" ) );
        std::unique_ptr<energy_observer>   observer;
        std::unique_ptr<physical_observer> physical;
        if ( record )
        {
            physical           = std::make_unique<physical_observer>( ops_, model_ );
            const auto initial = physical->evaluate( *state );
            history_.push_back(
                { { "time", 0 },
                  { "energy", initial.energy },
                  { "mass", initial.mass },
                  { "physical_dissipation", initial.dissipation } }
            );
            if ( method == "IMEX_EULER" )
                observer = std::make_unique<energy_observer>( ops_, model_ );
            save( 0, 0, 0, *state, initial );
        }
        std::size_t    warmup        = 0;
        const double   mass          = model_->cell_area() * ops_->sum( *state );
        double         time          = 0;
        const unsigned budget        = Adaptive ? config_.at( "maximum_steps" ).get<unsigned>() : count;
        const double   time_roundoff = 8 * std::numeric_limits<double>::epsilon() * budget * std::max( 1., end );
        unsigned       accepted = 0, rejected = 0, maximum_order = 0;
        double         minimum_step = end, maximum_step = 0;
        json           order_counts = json::array( { 0, 0, 0, 0, 0, 0 } );
        for ( unsigned k = 0; k < budget && ( !Adaptive || time < end - time_roundoff ); ++k )
        {
            ctx.step->set_time( time );
            ctx.step->set_target_time( end );
            ctx.step->apply( *state, *out );
            if ( ctx.step->get_status() != single_step_status::converged )
            {
                std::cerr << method << " stage failed at t=" << time << ", step=" << k + 1
                          << ", dt=" << ctx.step->get_dt();
                if ( ctx.linear->calls )
                {
                    const auto &monitor = ctx.linear->core.monitor();
                    std::cerr << "; last GMRES iterations=" << monitor.iters_performed()
                              << ", residual=" << monitor.resid_norm_out() << ", tolerance=" << monitor.tol_out();
                }
                std::cerr << '\n';
                throw std::runtime_error( "CH benchmark stage failed" );
            }
            check( ops_->is_valid_number( *out ), "CH benchmark nonfinite candidate" );
            if constexpr ( !Adaptive )
                check(
                    std::abs( ctx.step->get_dt() - h ) <= time_roundoff,
                    "CH benchmark did not reach its scheduled endpoint"
                );
            // Committing a separately rounded k*h can mark an ordinary BDF
            // endpoint as modified and discard history. Carry the actual endpoint.
            const double used     = ctx.step->get_dt();
            const double end_time = used == end - time ? end : time + used;
            check(
                used > 0 && end_time > time && used <= h + time_roundoff && end_time <= end + time_roundoff,
                "Invalid CH accepted step"
            );
            const unsigned order = [&] {
                if constexpr ( E == engine::bdf )
                    return ctx.step->get_order();
                else
                    return method_order( method );
            }();
            const unsigned attempts = ctx.step->get_attempts();
            check( order >= 1 && order <= 5 && attempts >= 1, "Invalid CH order/attempt count" );
            ++accepted;
            rejected += attempts - 1;
            order_counts[order] = order_counts[order].get<unsigned>() + 1;
            maximum_order       = std::max( maximum_order, order );
            minimum_step        = std::min( minimum_step, used );
            maximum_step        = std::max( maximum_step, used );
            if ( record )
            {
                const auto endpoint = physical->evaluate( *out );
                check( std::abs( endpoint.mass - mass ) < 2e-10, "CH benchmark lost mass" );
                json row = {
                    { "time", end_time },
                    { "step", ctx.step->get_dt() },
                    { "energy", endpoint.energy },
                    { "mass", endpoint.mass },
                    { "physical_dissipation", endpoint.dissipation },
                    { "order", order },
                    { "attempts", attempts }
                };
                if ( observer )
                {
                    const auto r = observer->evaluate_step( *state, *out, ctx.step->get_dt() );
                    check(
                        r.step_defect_norm < 2e-10 && std::abs( r.residual_work ) < 2e-10,
                        "CH benchmark inaccurate nonlinear solve"
                    );
                    check(
                        r.dissipation >= -r.roundoff_budget &&
                            r.dissipation_bound_defect <= r.residual_work + r.roundoff_budget &&
                            r.energy_after <= r.energy_before + std::abs( r.residual_work ) + r.roundoff_budget,
                        "CH benchmark energy inequality violated"
                    );
                    check( std::abs( r.mass_after - mass ) < 2e-10, "CH benchmark lost mass" );
                    row.update(
                        { { "dissipation", r.dissipation },
                          { "residual_work", r.residual_work },
                          { "bound_defect", r.dissipation_bound_defect },
                          { "roundoff_budget", r.roundoff_budget },
                          { "step_defect_rms", r.step_defect_norm } }
                    );
                }
                history_.push_back( row );
                if constexpr ( Adaptive )
                {
                    while ( samples_.size() < snapshots_.size() &&
                            sample_time( samples_.size() ) <= end_time + time_roundoff )
                    {
                        const auto   frame     = samples_.size();
                        const double target    = sample_time( frame );
                        const double theta     = std::clamp( ( target - time ) / used, 0., 1. );
                        const auto   view      = ctx.step->get_continuous_integration();
                        const double evaluated = view.evaluate( theta, **snapshots_[frame] );
                        check( std::abs( evaluated - target ) <= time_roundoff, "CH dense sample time mismatch" );
                        append_sample( accepted, target, evaluated, physical->evaluate( **snapshots_[frame] ) );
                    }
                }
                else
                {
                    const unsigned frame = samples_.size();
                    if ( frame < snapshots_.size() && k + 1 == frame * count / ( snapshots_.size() - 1 ) )
                        save( accepted, sample_time( frame ), end_time, *out, endpoint );
                }
            }
            ctx.step->finalize( adaptation_status::accepted, end_time, *out );
            time = end_time;
            ops_->assign( *out, *state );
            if ( k == 0 )
                warmup = guarded_memory::allocation_count;
            check( warmup == guarded_memory::allocation_count, "CH benchmark allocated backend storage after warmup" );
        }
        check( std::abs( time - end ) <= time_roundoff, "CH benchmark final time mismatch" );
        ops_->assign( *state, result );
        json summary = {
            { "steps", accepted },
            { "step", Adaptive ? json( nullptr ) : json( h ) },
            { "minimum_step", minimum_step },
            { "maximum_step", maximum_step },
            { "maximum_order", maximum_order },
            { "order_counts", order_counts },
            { "rejected_attempts", rejected },
            { "backend_allocations_after_warmup", 0 }
        };
        if constexpr ( Adaptive )
        {
            summary["relative_tolerance"] = config_.at( "relative_tolerance" ).get<double>() * scale;
            summary["absolute_tolerance"] = config_.at( "absolute_tolerance" ).get<double>() * scale;
        }
        return summary;
    }

    template <engine E>
    json select_mode( unsigned count, double end, vector &result, const std::string &method, bool record, double scale )
    {
        if ( config_.value( "adaptive", false ) || method == "BDF" )
            return integrate<E, true>( count, end, result, method, record, scale );
        return integrate<E>( count, end, result, method, record );
    }

    json selected( unsigned count, double end, vector &result, bool record = false, double scale = 1 )
    {
        const std::string method = config_.value( "method", "IMEX_EULER" );
        if ( method.rfind( "IMEX_", 0 ) == 0 )
            return select_mode<engine::imex>( count, end, result, method, record, scale );
        if ( method.rfind( "BDF", 0 ) == 0 )
            return select_mode<engine::bdf>( count, end, result, method, record, scale );
        if ( method.rfind( "ROS", 0 ) == 0 )
            return select_mode<engine::rosenbrock>( count, end, result, method, record, scale );
        return select_mode<engine::dirk>( count, end, result, method, record, scale );
    }

    json run()
    {
        const std::string method   = config_.value( "method", "IMEX_EULER" );
        const bool        adaptive = config_.value( "adaptive", false ) || method == "BDF";
        const double      end = config_.at( "end" ), requested = config_.at( "step" );
        const unsigned    count = unsigned( std::ceil( end / requested - 1e-12 ) );
        buffer<space>     endpoint( *ops_ ), reference( *ops_ ), coarse_reference( *ops_ ), scratch( *ops_ );
        for ( auto *value : { &endpoint, &reference, &coarse_reference, &scratch } )
            value->start_use();
        const auto     main = selected( count, end, *endpoint, true );
        const unsigned base = config_.at( "base_steps" ), levels = config_.at( "levels" );
        const unsigned finest    = base * ( 1u << ( levels - 1 ) );
        const double   order_end = config_.at( "order_end" );
        auto           coarse    = integrate<engine::rosenbrock>( finest * 8, order_end, *coarse_reference, "ROS3" );
        auto           fine      = integrate<engine::rosenbrock>( finest * 16, order_end, *reference, "ROS3" );
        const double   reference_difference = difference( *ops_, *coarse_reference, *reference, *scratch );
        json           refinements          = json::array();
        for ( unsigned level = 0; level < levels; ++level )
        {
            auto row =
                selected( adaptive ? 1u : base * ( 1u << level ), order_end, *endpoint, false, std::pow( .1, level ) );
            row["error_rms"] = difference( *ops_, *endpoint, *reference, *scratch );
            refinements.push_back( row );
        }

        // The transfer guard stays enabled for EVERY trajectory and diagnostic.
        // Only bounded, saved fields are exported, after all solves have ended.
        json                fields = json::array();
        std::vector<double> host( model_->grid.size() );
        check( samples_.size() == snapshots_.size(), "Incomplete CH field sampling" );
        for ( std::size_t i = 0; i < samples_.size(); ++i )
        {
            guarded_memory::check_transfers = false;
            try
            {
                ops_->get( **snapshots_[i], host.data() );
            }
            catch ( ... )
            {
                guarded_memory::check_transfers = true;
                throw;
            }
            guarded_memory::check_transfers = true;
            auto frame                      = samples_[i];
            frame["concentration"]          = host;
            fields.push_back( frame );
        }
        return {
            { "format", "ch.energy.visualization.v3" },
            { "configuration", config_ },
            { "method", method },
            { "expected_order", adaptive ? json( nullptr ) : json( method_order( method ) ) },
            { "primary_order", method_order( method ) ? json( method_order( method ) ) : json( nullptr ) },
            { "validation_mode", adaptive ? "tolerance_refinement" : "temporal_refinement" },
            { "energy_policy",
              config_.value( "method", "IMEX_EULER" ) == "IMEX_EULER" ? "convex_split_euler" : "observe_only" },
            { "norm", "physical_concentration_rms" },
            { "fit_error_floor", 1e-11 },
            { "trajectory", main },
            { "history", history_ },
            { "refinements", refinements },
            { "snapshots", fields },
            { "reference",
              { { "method", "ROS3" },
                { "coarse", coarse },
                { "fine", fine },
                { "difference_rms", reference_difference } } },
            { "integration_bulk_host_transfer_bytes", 0 },
            { "export_bulk_host_transfer_bytes", fields.size() * model_->grid.size() * sizeof( double ) }
        };
    }

private:
    double sample_time( std::size_t frame ) const
    {
        const double end = config_.at( "end" );
        if ( frame + 1 == config_.at( "frames" ).get<std::size_t>() )
            return end;
        const auto count = std::size_t( std::ceil( end / config_.at( "step" ).get<double>() - 1e-12 ) );
        return double( frame * count / ( config_.at( "frames" ).get<std::size_t>() - 1 ) ) * ( end / count );
    }
    void append_sample(
        unsigned index, double target, double evaluated, const ch::physical_diagnostics<space>::report_type &r
    )
    {
        samples_.push_back(
            { { "source_step_index", index },
              { "time", target },
              { "evaluation_time", evaluated },
              { "energy", r.energy },
              { "mass", r.mass },
              { "physical_dissipation", r.dissipation } }
        );
    }
    void save(
        unsigned index, double target, double evaluated, const vector &state,
        const ch::physical_diagnostics<space>::report_type &r
    )
    {
        // All snapshots are allocated before the first step; subsequent calls only copy backend-to-backend.
        if ( snapshots_.empty() )
        {
            const unsigned frames = config_.at( "frames" );
            for ( unsigned i = 0; i < frames; ++i )
            {
                auto value = std::make_unique<buffer<space>>( *ops_ );
                value->start_use();
                snapshots_.push_back( std::move( value ) );
            }
        }
        ops_->assign( state, **snapshots_.at( samples_.size() ) );
        append_sample( index, target, evaluated, r );
    }
    json                                        config_, history_ = json::array();
    std::shared_ptr<const model_type>           model_;
    std::shared_ptr<space>                      ops_;
    std::vector<std::unique_ptr<buffer<space>>> snapshots_;
    json                                        samples_ = json::array();
};

int main( int argc, char **argv )
try
{
#ifdef CH_FOURIER
    if ( argc == 2 && std::string( argv[1] ) == "--verify-spatial" )
    {
        backend_type::init_device();
        ch_fourier_tests::verify();
        finish();
        return 0;
    }
#endif
    if ( argc != 3 )
        throw std::invalid_argument( "Usage: ch_energy_benchmark config.json report.json" );
    std::ifstream input( argv[1] );
    const auto    config = json::parse( input );
    check( config.value( "spatial", "finite_difference" ) == spatial_method, "Wrong CH spatial benchmark binary" );
    method_order( config.value( "method", "IMEX_EULER" ) );
    const unsigned grid = config.at( "grid" ), levels = config.at( "levels" ), base = config.at( "base_steps" ),
                   frames = config.at( "frames" );
    check(
        grid >= 8 && grid <= 64 && levels >= 3 && levels <= 6 && base >= 4 && base <= 64 && frames >= 2 && frames <= 9,
        "Invalid CH benchmark grid/refinement/storage budget"
    );
    for ( const char *key : { "end", "step", "order_end", "mobility", "epsilon", "amplitude" } )
        check(
            std::isfinite( config.at( key ).get<double>() ) && config.at( key ).get<double>() > 0,
            "Invalid CH benchmark parameter"
        );
    check( std::isfinite( config.at( "mean" ).get<double>() ), "Invalid CH mean" );
    const double steps = std::ceil( config.at( "end" ).get<double>() / config.at( "step" ).get<double>() - 1e-12 );
    check( steps >= frames - 1 && steps <= 20000, "Invalid CH benchmark step budget" );
    const std::string method = config.value( "method", "IMEX_EULER" );
    if ( config.value( "adaptive", false ) || method == "BDF" )
    {
        check(
            method == "BDF" || method == "ARK324L2SA" || method == "ROS34PW2" || method == "IMEX_ARK324L2SA",
            "Unsupported CH adaptive method"
        );
        for ( const char *key : { "relative_tolerance", "absolute_tolerance", "initial_step" } )
            check(
                std::isfinite( config.at( key ).get<double>() ) && config.at( key ).get<double>() > 0,
                "Invalid CH adaptive parameter"
            );
        check(
            config.at( "maximum_steps" ).get<unsigned>() >= 1 &&
                config.at( "maximum_steps" ).get<unsigned>() <= 20000 &&
                config.at( "maximum_order" ).get<unsigned>() >= 1 &&
                config.at( "maximum_order" ).get<unsigned>() <= 5 &&
                config.at( "relative_tolerance" ).get<double>() * std::pow( .1, levels - 1 ) >=
                    100 * std::numeric_limits<double>::epsilon(),
            "Invalid CH adaptive budget/refinement floor"
        );
    }
    backend_type::init_device();
    const auto report = experiment( config ).run();
    check( guarded_memory::allocations.empty(), "CH benchmark leaked backend allocations" );
    std::ofstream output( argv[2] );
    output.exceptions( std::ios::badbit | std::ios::failbit );
    output << report.dump( 2 ) << '\n';
    std::cout << "CH: exported " << report.at( "snapshots" ).size() << " snapshots after integration\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
