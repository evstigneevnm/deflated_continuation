#ifndef NMFD_TIME_STEPPERS_BUTCHER_TABLE_H
#define NMFD_TIME_STEPPERS_BUTCHER_TABLE_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{

class butcher_table
{
public:
    using scalar_type = long double;
    using vector_type = std::vector<scalar_type>;
    using matrix_type = std::vector<vector_type>;

    enum class scheme_type
    {
        explicit_rk,
        dirk,
        sdirk,
        irk
    };

    enum class error_estimator_type
    {
        none,
        embedded_pair,
        dop853_combined
    };

    butcher_table(
        matrix_type a, vector_type b, unsigned int order, vector_type c = {}, vector_type embedded_b = {},
        unsigned int embedded_order = 0, scalar_type coefficient_tolerance = 2e-15L,
        matrix_type dense_coefficients = {}, unsigned int dense_order = 0, matrix_type dense_outout_a = {},
        vector_type dense_outout_c = {}, matrix_type error_coefficients = {}
    )
        : a_( std::move( a ) ), b_( std::move( b ) ), c_( std::move( c ) ), embedded_b_( std::move( embedded_b ) ),
          order_( order ), embedded_order_( embedded_order ), tolerance_( coefficient_tolerance ),
          dense_coefficients_( std::move( dense_coefficients ) ), dense_order_( dense_order ),
          dense_outout_a_( std::move( dense_outout_a ) ), dense_outout_c_( std::move( dense_outout_c ) ),
          error_coefficients_( std::move( error_coefficients ) )
    {
        if ( b_.empty() || a_.size() != b_.size() || order_ == 0 || !std::isfinite( tolerance_ ) || tolerance_ <= 0 ||
             embedded_b_.empty() != ( embedded_order_ == 0 ) || ( !embedded_b_.empty() && embedded_order_ >= order_ ) )
        {
            throw std::invalid_argument( "Invalid Butcher table dimensions/order/tolerance" );
        }
        for ( const auto &row : a_ )
        {
            validate_vector( row, size() );
        }
        validate_vector( b_, size() );
        check_equal( std::accumulate( b_.begin(), b_.end(), 0.L ), 1.L );
        if ( c_.empty() )
        {
            for ( const auto &row : a_ )
            {
                c_.push_back( std::accumulate( row.begin(), row.end(), 0.L ) );
            }
        }
        validate_vector( c_, size() );
        for ( std::size_t i = 0; i < size(); ++i )
        {
            check_equal( c_[i], std::accumulate( a_[i].begin(), a_[i].end(), 0.L ) );
        }
        if ( is_embedded() )
        {
            validate_vector( embedded_b_, size() );
            check_equal( std::accumulate( embedded_b_.begin(), embedded_b_.end(), 0.L ), 1.L );
        }
        if ( !error_coefficients_.empty() )
        {
            if ( is_embedded() || order_ != 8 || error_coefficients_.size() != 2 )
            {
                throw std::invalid_argument( "Invalid DOP853 combined estimator" );
            }
            for ( const auto &row : error_coefficients_ )
            {
                validate_vector( row, size() );
                check_equal( std::accumulate( row.begin(), row.end(), 0.L ), 0.L );
            }
        }
        if ( dense_coefficients_.empty() != ( dense_order_ == 0 ) || dense_order_ > order_ )
        {
            throw std::invalid_argument( "Invalid dense-output order" );
        }
        if ( dense_outout_a_.size() != dense_outout_c_.size() || ( !has_dense_output() && !dense_outout_a_.empty() ) )
        {
            throw std::invalid_argument( "Invalid additional dense-output stages" );
        }
        validate_vector( dense_outout_c_, dense_outout_a_.size() );
        for ( std::size_t i = 0; i < dense_outout_a_.size(); ++i )
        {
            const auto &row = dense_outout_a_[i];
            validate_vector( row, dense_outout_stage_count() );
            check_equal( std::accumulate( row.begin(), row.end(), 0.L ), dense_outout_c_[i] );
            for ( std::size_t j = size() + i; j < row.size(); ++j )
            {
                if ( row[j] != 0 )
                {
                    throw std::invalid_argument( "Dense-output stages must use earlier derivatives only" );
                }
            }
        }
        if ( has_dense_output() )
        {
            if ( dense_coefficients_.size() != dense_outout_stage_count() || dense_degree() < dense_order_ )
            {
                throw std::invalid_argument( "Invalid dense-output dimensions" );
            }
            for ( std::size_t i = 0; i < dense_outout_stage_count(); ++i )
            {
                validate_vector( dense_coefficients_[i], dense_degree() );
                check_equal( dense_b( i, 1 ), i < size() ? b_[i] : 0.L );
            }
            for ( std::size_t j = 0; j < dense_degree(); ++j )
            {
                scalar_type sum = 0;
                for ( const auto &row : dense_coefficients_ )
                {
                    sum += row[j];
                }
                check_equal( sum, j == 0 ? 1.L : 0.L );
            }
        }
    }

    std::size_t size() const
    {
        return b_.size();
    }

    unsigned int order() const
    {
        return order_;
    }

    unsigned int embedded_order() const
    {
        return embedded_order_;
    }

    unsigned int error_order() const
    {
        return error_estimator() == error_estimator_type::dop853_combined ? 8 : is_embedded() ? embedded_order_ + 1 : 0;
    }

    bool is_embedded() const
    {
        return !embedded_b_.empty();
    }

    error_estimator_type error_estimator() const
    {
        return !error_coefficients_.empty() ? error_estimator_type::dop853_combined
               : is_embedded()              ? error_estimator_type::embedded_pair
                                            : error_estimator_type::none;
    }

    bool has_error_estimate() const
    {
        return error_estimator() != error_estimator_type::none;
    }

    scalar_type coefficient_tolerance() const
    {
        return tolerance_;
    }

    scalar_type a( std::size_t i, std::size_t j ) const
    {
        return a_.at( i ).at( j );
    }

    scalar_type b( std::size_t i ) const
    {
        return b_.at( i );
    }

    scalar_type c( std::size_t i ) const
    {
        return c_.at( i );
    }

    scalar_type embedded_b( std::size_t i ) const
    {
        return embedded_b_.at( i );
    }

    scalar_type error_b( std::size_t i ) const
    {
        return error_coefficients_.empty() ? b( i ) - embedded_b( i ) : error_coefficients_[0].at( i );
    }

    scalar_type secondary_error_b( std::size_t i ) const
    {
        return error_coefficients_.at( 1 ).at( i );
    }

    bool has_dense_output() const
    {
        return dense_order_ != 0;
    }

    unsigned int dense_order() const
    {
        return dense_order_;
    }

    // Total derivatives used by the interpolant, including the ordinary stages.
    std::size_t dense_outout_stage_count() const
    {
        return size() + dense_outout_a_.size();
    }

    scalar_type dense_outout_a( std::size_t i, std::size_t j ) const
    {
        if ( j >= dense_outout_stage_count() )
        {
            throw std::out_of_range( "Dense-output stage index" );
        }
        if ( i < size() )
        {
            return j < size() ? a( i, j ) : 0.L;
        }
        return dense_outout_a_.at( i - size() ).at( j );
    }

    scalar_type dense_outout_c( std::size_t i ) const
    {
        return i < size() ? c( i ) : dense_outout_c_.at( i - size() );
    }

    std::size_t dense_degree() const
    {
        return dense_coefficients_.empty() ? 0 : dense_coefficients_[0].size();
    }

    // b_i(theta) = sum_j P_ij theta^(j+1), not the embedded weights.
    scalar_type dense_coefficient( std::size_t i, std::size_t j ) const
    {
        return dense_coefficients_.at( i ).at( j );
    }

    scalar_type dense_b( std::size_t i, scalar_type theta ) const
    {
        scalar_type value = 0;
        const auto &row   = dense_coefficients_.at( i );
        for ( auto j = row.size(); j > 0; --j )
        {
            value = theta * value + row[j - 1];
        }
        return theta * value;
    }

    scheme_type type() const
    {
        // Structural zeros are exact: a tiny nonzero upper entry is still implicit.
        for ( std::size_t i = 0; i < size(); ++i )
        {
            for ( std::size_t j = i + 1; j < size(); ++j )
            {
                if ( a_[i][j] != 0 )
                {
                    return scheme_type::irk;
                }
            }
        }
        bool zero_diagonal = true, same_diagonal = true;
        for ( std::size_t i = 0; i < size(); ++i )
        {
            zero_diagonal = zero_diagonal && a_[i][i] == 0;
            same_diagonal = same_diagonal && a_[i][i] == a_[0][0];
        }
        if ( zero_diagonal )
        {
            return scheme_type::explicit_rk;
        }
        return same_diagonal ? scheme_type::sdirk : scheme_type::dirk;
    }

private:
    matrix_type  a_;
    vector_type  b_, c_, embedded_b_;
    unsigned int order_, embedded_order_;
    scalar_type  tolerance_;
    matrix_type  dense_coefficients_;
    unsigned int dense_order_;
    matrix_type  dense_outout_a_;
    vector_type  dense_outout_c_;
    matrix_type  error_coefficients_;

    static void validate_vector( const vector_type &v, std::size_t size )
    {
        if ( v.size() != size )
        {
            throw std::invalid_argument( "Butcher table shape mismatch" );
        }
        for ( const auto value : v )
        {
            if ( !std::isfinite( value ) )
            {
                throw std::invalid_argument( "Nonfinite Butcher coefficient" );
            }
        }
    }

    void check_equal( scalar_type x, scalar_type y ) const
    {
        if ( std::abs( x - y ) > tolerance_ * std::max( { 1.L, std::abs( x ), std::abs( y ) } ) )
        {
            throw std::invalid_argument( "Inconsistent Butcher weights or stage times" );
        }
    }
};
}
}
}
#endif
