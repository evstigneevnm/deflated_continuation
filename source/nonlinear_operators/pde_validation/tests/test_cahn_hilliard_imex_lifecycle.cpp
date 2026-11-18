#include "cahn_hilliard_imex_test_support.h"
using namespace ch_tests;

// Fault injection belongs to test assembly. Direct CH kernels have no iterative
// mass solve; bool still denotes successful rate evaluation, not energy decay.
template <class S>
class faulty_explicit : public ch::explicit_operator<S>
{
    std::shared_ptr<S> ops_;

public:
    using vector_type = typename S::vector_type;
    unsigned fault    = 0;
    faulty_explicit( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model )
        : ch::explicit_operator<S>( ops, model ), ops_( std::move( ops ) )
    {
    }
    bool evaluate_rate( const vector_type &in, vector_type &out ) const
    {
        if ( fault == 1 )
            return false;
        if ( fault == 2 )
        {
            ops_->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
            return true;
        }
        if ( fault == 3 )
            throw std::runtime_error( "Injected CH rate exception" );
        return ch::explicit_operator<S>::evaluate_rate( in, out );
    }
};
template <class S>
struct faulty_problem
{
    using imex_operator_type = ts::runge_kutta::imex_operator<faulty_explicit<S>, ch::implicit_operator<S>>;
    std::shared_ptr<imex_operator_type> bundle;
    faulty_problem( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model )
        : bundle(
              std::make_shared<imex_operator_type>(
                  std::make_shared<faulty_explicit<S>>( ops, model ),
                  std::make_shared<ch::implicit_operator<S>>( ops, model )
              )
          )
    {
    }
    std::shared_ptr<imex_operator_type> get_imex_operator() const
    {
        return bundle;
    }
};
void failures()
{
    auto               model = std::make_shared<const model_type>( 8 );
    auto               ops   = std::make_shared<space_type>( model->grid.size() );
    buffer<space_type> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), first( *ops ), scratch( *ops );
    for ( auto *b : { &state, &initial, &out, &sentinel, &first, &scratch } )
        b->start_use();
    for ( unsigned fault : { 1, 2, 3, 4 } )
        for ( bool alias : { false, true } )
        {
            context<space_type, faulty_problem<space_type>> ctx( ops, model, .04 );
            auto                                            bundle = ctx.problem->get_imex_operator();
            bundle->explicit_part->fault                           = fault == 4 ? 0 : fault;
            ctx.linear->fail_on                                    = fault == 4 ? 1 : 0;
            initialize( *ops, *model, *state );
            ops->assign( *state, *initial );
            ops->assign_scalar( 7., *out );
            ops->assign( *out, *sentinel );
            ctx.step->set_time( .3 );
            ctx.step->set_target_time( .34 );
            bool caught = false;
            try
            {
                ctx.step->apply( *state, alias ? *state : *out );
            }
            catch ( const std::runtime_error & )
            {
                caught = true;
            }
            check( caught == ( fault == 3 ), "CH exception handling changed" );
            check( ctx.step->get_status() != single_step_status::converged, "CH failed stage reported success" );
            check( difference( *ops, *state, *initial, *scratch ) == 0, "CH failure changed committed input" );
            check( difference( *ops, *out, *sentinel, *scratch ) == 0, "CH failure published output" );
            bundle->explicit_part->fault = 0;
            ctx.linear->fail_on          = 0;
            ctx.step->reset();
            ctx.control->reset();
            double h = apply( ctx, *initial, *out, .3 );
            ops->assign( *out, *first );
            ctx.step->finalize( adaptation_status::accepted, .3 + h, *out );
            ctx.step->reset();
            ctx.control->reset();
            apply( ctx, *initial, *out, .3 );
            check( difference( *ops, *first, *out, *scratch ) < 1e-12, "CH restart retained failed stage state" );
        }
}
int main()
try
{
    backend_type::init_device();
    failures();
    auto                               model = std::make_shared<const model_type>( 8 );
    auto                               ops   = std::make_shared<space_type>( model->grid.size() );
    ch::energy_diagnostics<space_type> observer( ops, model );
    buffer<space_type>                 x( *ops );
    x.start_use();
    initialize( *ops, *model, *x );
    for ( double h : { 0., -.1, std::numeric_limits<double>::infinity() } )
    {
        bool rejected = false;
        try
        {
            observer.evaluate_step( *x, *x, h );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        check( rejected, "CH energy observer accepted an invalid step" );
    }
    ops->assign_scalar( 1e100, *x );
    bool rejected = false;
    try
    {
        observer.evaluate_step( *x, *x, .1 );
    }
    catch ( const std::overflow_error & )
    {
        rejected = true;
    }
    check( rejected, "CH energy observer accepted overflowed diagnostics" );
    using Problem = ch::convex_split_problem<space_type>;
    std::weak_ptr<typename Problem::imex_operator_type> weak;
    {
        auto ctx = std::make_unique<context<space_type>>( ops, model, .04 );
        weak     = ctx->problem->get_imex_operator();
        ctx->problem.reset();
        check( !weak.expired(), "CH timestepper lost its partition bundle" );
    }
    check( weak.expired(), "CH partition ownership cycle" );
    finish();
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
