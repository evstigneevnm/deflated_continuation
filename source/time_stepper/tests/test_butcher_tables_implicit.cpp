#include "table_order_checks.h"

using namespace nmfd::time_steppers::runge_kutta;
using table_tests::require;

int main()
try
{
    const struct
    {
        const char  *name;
        std::size_t  stages;
        unsigned int order, embedded_order;
    } methods[] = { { "IE", 1, 1, 0 },         { "IM", 1, 2, 0 },          { "CN", 2, 2, 0 },
                    { "SDIRK2(1)2", 2, 2, 1 }, { "ESDIRK2(1)3", 3, 2, 1 }, { "SDIRK3(1)3", 3, 3, 1 } };

    for ( const auto &method : methods )
    {
        const auto *name  = method.name;
        const auto  table = make_butcher_table( name );
        require(
            table.size() == method.stages && table.order() == method.order &&
                table.embedded_order() == method.embedded_order,
            std::string( name ) + ": name/metadata mismatch"
        );
        require( table.type() != butcher_table::scheme_type::explicit_rk, name );
        table_tests::check_order( table, name );
        if ( table.is_embedded() )
        {
            table_tests::check_order( table, name, true );
        }
    }
    require( make_butcher_table( "CN" ).type() == butcher_table::scheme_type::dirk, "CN classification" );
    require( make_butcher_table( "SDIRK3(1)3" ).error_order() == 2, "SDIRK3(1)3 error scales as h^2" );

    const struct
    {
        const char  *name;
        std::size_t  stored_stages;
        unsigned int order;
    } pairs[] = { { "IMEX_EULER", 2, 1 }, { "IMEX_HEUN_TR2", 2, 2 }, { "IMEX_ARS233", 3, 3 }, { "IMEX_ARS222", 3, 2 } };

    for ( const auto &method : pairs )
    {
        const auto *name = method.name;
        const auto  pair = make_imex_butcher_table( name );
        const auto &e    = pair.explicit_table;
        const auto &i    = pair.implicit_table;
        require(
            pair.order == method.order && e.size() == method.stored_stages && i.size() == method.stored_stages,
            std::string( name ) + ": name/metadata mismatch"
        );
        table_tests::check_order( e, std::string( name ) + " explicit" );
        table_tests::check_order( i, std::string( name ) + " implicit" );
        // All colored rooted-tree conditions through order three.
        for ( const auto *root : { &e, &i } )
        {
            for ( const auto *child : { &e, &i } )
            {
                long double bc = 0, bcc = 0;
                for ( std::size_t j = 0; j < e.size(); ++j )
                {
                    bc += root->b( j ) * child->c( j );
                    bcc += root->b( j ) * child->c( j ) * child->c( j );
                }
                if ( pair.order >= 2 )
                {
                    require( std::abs( bc - .5L ) < 1e-14L, "IMEX coupling order two" );
                }
                if ( pair.order >= 3 )
                {
                    require( std::abs( bcc - 1.L / 3 ) < 1e-14L, "IMEX bush order three" );
                    for ( const auto *grandchild : { &e, &i } )
                    {
                        long double bac = 0;
                        for ( std::size_t j = 0; j < e.size(); ++j )
                        {
                            for ( std::size_t k = 0; k < e.size(); ++k )
                            {
                                bac += root->b( j ) * child->a( j, k ) * grandchild->c( k );
                            }
                        }
                        require( std::abs( bac - 1.L / 6 ) < 1e-14L, "IMEX chain order three" );
                    }
                }
            }
        }
    }
    bool rejected = false;
    try
    {
        imex_butcher_table bad( make_butcher_table( "HE" ), make_butcher_table( "SDIRK2(1)2" ), 2 );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Mismatched IMEX stage times" );
    std::cout << "Implicit/IMEX tables: PASS (no implicit stepper in this test)\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
