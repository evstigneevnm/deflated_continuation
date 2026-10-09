#ifndef NMFD_TIME_STEPPERS_BUTCHER_TABLES_H
#define NMFD_TIME_STEPPERS_BUTCHER_TABLES_H

#include <string>
#include <time_stepper/runge_kutta/butcher_table.h>
#include <time_stepper/runge_kutta/dop853_coefficients.h>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{

// DIRK names use p(q)s: primary order p, embedded order q, total stages s.
inline butcher_table make_butcher_table( const std::string &name )
{
    if ( name == "EE" )
    {
        return { { { 0 } }, { 1 }, 1, {}, {}, 0, 2e-15L, { { 1 } }, 1 };
    }
    if ( name == "HE" )
    {
        return { { { 0, 0 }, { 1, 0 } }, { .5L, .5L }, 2, {}, { 1, 0 }, 1, 2e-15L, { { 1, -.5L }, { 0, .5L } }, 2 };
    }
    if ( name == "RK23" )
    {
        // Bogacki-Shampine 3(2), including the endpoint derivative and native cubic extension.
        return {
            { { 0, 0, 0, 0 }, { 1.L / 2, 0, 0, 0 }, { 0, 3.L / 4, 0, 0 }, { 2.L / 9, 1.L / 3, 4.L / 9, 0 } },
            { 2.L / 9, 1.L / 3, 4.L / 9, 0 },
            3,
            { 0, 1.L / 2, 3.L / 4, 1 },
            { 7.L / 24, 1.L / 4, 1.L / 3, 1.L / 8 },
            2,
            2e-15L,
            { { 1, -4.L / 3, 5.L / 9 }, { 0, 1, -2.L / 3 }, { 0, 4.L / 3, -8.L / 9 }, { 0, -1, 1 } },
            3
        };
    }
    if ( name == "RK33SSP" )
    {
        return {
            { { 0, 0, 0 }, { 1, 0, 0 }, { .25L, .25L, 0 } },
            { 1.L / 6, 1.L / 6, 2.L / 3 },
            3,
            {},
            { .291485418878409L, .291485418878409L, .417029162243181L },
            2
        };
    }
    if ( name == "RK43SSP" )
    {
        return {
            { { 0, 0, 0, 0 }, { .5L, 0, 0, 0 }, { .5L, .5L, 0, 0 }, { 1.L / 6, 1.L / 6, 1.L / 6, 0 } },
            { 1.L / 6, 1.L / 6, 1.L / 6, .5L },
            3,
            {},
            { 1.L / 3, 1.L / 3, 1.L / 3, 0 },
            2
        };
    }
    if ( name == "RK45" )
    {
        return {
            { { 0, 0, 0, 0, 0, 0, 0 },
              { 1.L / 5, 0, 0, 0, 0, 0, 0 },
              { 3.L / 40, 9.L / 40, 0, 0, 0, 0, 0 },
              { 44.L / 45, -56.L / 15, 32.L / 9, 0, 0, 0, 0 },
              { 19372.L / 6561, -25360.L / 2187, 64448.L / 6561, -212.L / 729, 0, 0, 0 },
              { 9017.L / 3168, -355.L / 33, 46732.L / 5247, 49.L / 176, -5103.L / 18656, 0, 0 },
              { 35.L / 384, 0, 500.L / 1113, 125.L / 192, -2187.L / 6784, 11.L / 84, 0 } },
            { 35.L / 384, 0, 500.L / 1113, 125.L / 192, -2187.L / 6784, 11.L / 84, 0 },
            5,
            { 0, 1.L / 5, 3.L / 10, 4.L / 5, 8.L / 9, 1, 1 },
            { 5179.L / 57600, 0, 7571.L / 16695, 393.L / 640, -92097.L / 339200, 187.L / 2100, 1.L / 40 },
            4,
            // Shampine's quartic extension (1986); rational coefficients also used by SciPy RK45:
            // https://github.com/scipy/scipy/blob/v1.16.0/scipy/integrate/_ivp/rk.py
            2e-15L,
            { { 1, -8048581381.L / 2820520608, 8663915743.L / 2820520608, -12715105075.L / 11282082432 },
              { 0, 0, 0, 0 },
              { 0, 131558114200.L / 32700410799, -68118460800.L / 10900136933, 87487479700.L / 32700410799 },
              { 0, -1754552775.L / 470086768, 14199869525.L / 1410260304, -10690763975.L / 1880347072 },
              { 0, 127303824393.L / 49829197408, -318862633887.L / 49829197408, 701980252875.L / 199316789632 },
              { 0, -282668133.L / 205662961, 2019193451.L / 616988883, -1453857185.L / 822651844 },
              { 0, 40617522.L / 29380423, -110615467.L / 29380423, 69997945.L / 29380423 } },
            4
        };
    }
    if ( name == "DOP853" )
    {
        return detail::make_dop853_table();
    }
    if ( name == "RK64SSP" )
    {
        // Published decimals: Fekete et al., JCAM 412 (2022) 114325, Table 4.
        // No long-double accuracy for this truncated table.
        return {
            { { 0, 0, 0, 0, 0, 0 },
              { .3552975516919L, 0, 0, 0, 0, 0 },
              { .2704882223931L, .3317866983600L, 0, 0, 0, 0 },
              { .1223997401356L, .1501381660925L, .1972127376054L, 0, 0, 0 },
              { .0763425067155L, .0936433683640L, .1230044665810L, .2718245927242L, 0, 0 },
              { .0763425067155L, .0936433683640L, .1230044665810L, .2718245927242L, .4358156542577L, 0 } },
            { .1522491819555L, .1867521364225L, .1555370561501L, .1348455085546L, .2161974490441L, .1544186678729L },
            4,
            {},
            { .1210663237182L, .2308844004550L, .0853424972752L, .3450614904457L, .0305351538213L, .1871101342844L },
            3,
            2e-12L
        };
    }
    if ( name == "IE" )
    {
        return { { { 1 } }, { 1 }, 1 };
    }
    if ( name == "IM" )
    {
        return { { { .5L } }, { 1 }, 2 };
    }
    if ( name == "CN" )
    {
        return { { { 0, 0 }, { .5L, .5L } }, { .5L, .5L }, 2 };
    }
    const auto gamma2 = 1.L - std::sqrt( 2.L ) / 2;
    if ( name == "SDIRK2(1)2" )
    {
        return { { { gamma2, 0 }, { 1 - gamma2, gamma2 } }, { 1 - gamma2, gamma2 }, 2, {}, { .5L, .5L }, 1 };
    }
    if ( name == "ESDIRK2(1)3" )
    {
        const auto g      = gamma2;
        const auto b2     = g * ( -2 + 7 * g - 5 * g * g + 4 * g * g * g ) / ( 2 * ( 2 * g - 1 ) );
        const auto b3     = -2 * g * g * ( 1 - g + g * g ) / ( 2 * g - 1 );
        const auto middle = ( 1 - 2 * g ) / ( 4 * g );
        return {
            { { 0, 0, 0 }, { g, g, 0 }, { 1 - middle - g, middle, g } },
            { 1 - middle - g, middle, g },
            2,
            {},
            { 1 - b2 - b3, b2, b3 },
            1
        };
    }
    if ( name == "SDIRK3(1)3" )
    {
        const auto g     = .43586652150845899941601945119356L;
        const auto alpha = 1 - 4 * g + 2 * g * g;
        const auto beta  = -1 + 6 * g - 9 * g * g + 3 * g * g * g;
        const auto b2    = -3 * alpha * alpha / ( 4 * beta );
        const auto c2    = ( 2 - 9 * g + 6 * g * g ) / ( 3 * alpha );
        // Alexander's third-order primary method with the inherited first-order estimator.
        return { { { g, 0, 0 }, { c2 - g, g, 0 }, { 1 - b2 - g, b2, g } },
                 { 1 - b2 - g, b2, g },
                 3,
                 {},
                 { .5L - b2, b2, .5L },
                 1 };
    }
    throw std::invalid_argument( "Unknown RK scheme: " + name );
}

struct imex_butcher_table
{
    butcher_table explicit_table, implicit_table;
    unsigned int  order;

    imex_butcher_table( butcher_table e, butcher_table i, unsigned int p )
        : explicit_table( std::move( e ) ), implicit_table( std::move( i ) ), order( p )
    {
        if ( explicit_table.type() != butcher_table::scheme_type::explicit_rk ||
             implicit_table.type() == butcher_table::scheme_type::irk ||
             explicit_table.size() != implicit_table.size() || p == 0 || p > explicit_table.order() ||
             p > implicit_table.order() )
        {
            throw std::invalid_argument( "Incompatible IMEX tables" );
        }
        for ( std::size_t j = 0; j < explicit_table.size(); ++j )
        {
            if ( std::abs( explicit_table.c( j ) - implicit_table.c( j ) ) >
                 std::max( explicit_table.coefficient_tolerance(), implicit_table.coefficient_tolerance() ) )
            {
                throw std::invalid_argument( "IMEX stage times differ" );
            }
        }
    }
};

// ARS names use the published (implicit stages, explicit stages, order) triplet,
// not the dimensions of the padded tableaux stored here.
inline imex_butcher_table make_imex_butcher_table( const std::string &name )
{
    if ( name == "IMEX_EULER" )
    {
        return { { { { 0, 0 }, { 1, 0 } }, { 1, 0 }, 1 }, { { { 0, 0 }, { 0, 1 } }, { 0, 1 }, 1 }, 1 };
    }
    if ( name == "IMEX_HEUN_TR2" )
    {
        return { { { { 0, 0 }, { 1, 0 } }, { .5L, .5L }, 2 }, { { { 0, 0 }, { .5L, .5L } }, { .5L, .5L }, 2 }, 2 };
    }
    if ( name == "IMEX_ARS233" )
    {
        const auto g = ( 3 + std::sqrt( 3.L ) ) / 6;
        return {
            { { { 0, 0, 0 }, { g, 0, 0 }, { g - 1, 2 * ( 1 - g ), 0 } }, { 0, .5L, .5L }, 3 },
            { { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - 2 * g, g } }, { 0, .5L, .5L }, 3 },
            3
        };
    }
    if ( name == "IMEX_ARS222" )
    {
        const auto g = 1.L - std::sqrt( 2.L ) / 2, k = 1 - 1 / ( 2 * g );
        return {
            { { { 0, 0, 0 }, { g, 0, 0 }, { k, 1 - k, 0 } }, { k, 1 - k, 0 }, 2 },
            { { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - g, g } }, { 0, 1 - g, g }, 2 },
            2
        };
    }
    throw std::invalid_argument( "Unknown IMEX scheme: " + name );
}
}
}
}
#endif
