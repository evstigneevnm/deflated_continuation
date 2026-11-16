#include <algorithm>
#include <iostream>
#include <nmfd/time_steppers/runge_kutta/butcher_tables.h>
#include <nmfd/time_steppers/rosenbrock/coefficients.h>
#include <nmfd/time_steppers/bdf/coefficients.h>

// Export actual coefficient metadata for the documentation contract check.
// Dense orders are differential orders, not guarantees for algebraic variables.
int main()
{
    using namespace nmfd::time_steppers;
    const auto row = []( const char *name, unsigned p, unsigned q, unsigned dense ) {
        std::cout << name << '|' << p << '|' << q << '|' << dense << '\n';
    };
    for ( const auto *name :
          { "EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK4", "RK64SSP", "RK45", "DOP853", "IE", "IM", "CN",
            "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
    {
        const auto t     = runge_kutta::make_butcher_table( name );
        const auto dense = t.has_dense_output() ? t.dense_order()
                           : t.type() == runge_kutta::butcher_table::scheme_type::explicit_rk
                               ? std::min( 3u, t.order() )
                               : 0;
        row( name, t.order(), t.embedded_order(), dense );
    }
    for ( const auto *name : { "IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS233", "IMEX_ARS222", "IMEX_ARK324L2SA" } )
    {
        const auto t = runge_kutta::make_imex_butcher_table( name );
        row( name, t.order, t.embedded_order(), t.has_dense_output() ? 3 : 0 );
    }
    for ( const auto *name : { "ROS2", "ROS3", "ROS34PW2", "ROSI2P2" } )
    {
        const rosenbrock::method_table t( name );
        row( name, t.order(), t.embedded_order(), t.dense_output_order() );
    }
    for ( unsigned q : { 1, 2 } )
    {
        const auto f = bdf::make_formula( q, .1, std::array<double, 5>{ .1, .1, .1, .1, .1 } );
        std::cout << "BDF" << q << '|' << f.order << "|0|" << f.order << '\n';
    }
    std::cout << "BDF|1-5|0|current\n";
}
