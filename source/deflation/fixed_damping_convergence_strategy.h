#ifndef __DEFLATION_FIXED_DAMPING_CONVERGENCE_STRATEGY_H__
#define __DEFLATION_FIXED_DAMPING_CONVERGENCE_STRATEGY_H__

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/vector_wrap.h>
#include <scfd/utils/logged_obj_base.h>

namespace deflation
{

// Legacy deflation acceptance, using NMFD's iteration and solver lifecycle.
template <class Space, class Log, class Operator, class Projector, class Quality>
class fixed_damping_convergence_strategy : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type       = typename Space::scalar_type;
    using vector_type       = typename Space::vector_type;
    using vector_space_type = Space;
    using logged_obj_type   = scfd::utils::logged_obj_base<Log>;

private:
    using T = scalar_type;

public:
    struct params : logged_obj_type::params
    {
        T            abs_tol = T( 1e-6 ), weight = T( .5 ), maximum_residual = T( 1e16 );
        unsigned int max_iters_num       = 1000;
        bool         store_norms_history = false, verbose = true;

        explicit params( const std::string &prefix = "", const std::string &name = "fixed_damping_deflation::" )
            : logged_obj_type::params( 0, prefix + name )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
            params updated           = *this;
            updated.abs_tol          = j.value( "abs_tol", abs_tol );
            updated.weight           = j.value( "weight", weight );
            updated.maximum_residual = j.value( "maximum_residual", maximum_residual );
            const auto limit         = j.find( "max_iters_num" );
            if ( limit != j.end() )
            {
                if ( !limit->is_number_integer() || *limit < 1 || *limit > std::numeric_limits<int>::max() )
                    throw std::invalid_argument( "Deflation iteration limit must be in [1, INT_MAX]" );
                updated.max_iters_num = limit->template get<unsigned int>();
            }
            updated.store_norms_history = j.value( "store_norms_history", store_norms_history );
            updated.verbose             = j.value( "verbose", verbose );
            validate( updated );
            *this = std::move( updated );
        }

        nlohmann::json to_json() const
        {
            return {
                { "abs_tol", abs_tol },
                { "weight", weight },
                { "maximum_residual", maximum_residual },
                { "max_iters_num", max_iters_num },
                { "store_norms_history", store_norms_history },
                { "verbose", verbose }
            };
        }
#endif
    };

    struct utils
    {
        std::shared_ptr<Space> vec_space;
        Log                   *log = nullptr;
        utils()                    = default;
        utils( std::shared_ptr<Space> space, Log *logger = nullptr ) : vec_space( std::move( space ) ), log( logger )
        {
        }
        template <class Backend>
        utils( Backend &backend, std::shared_ptr<Space> space ) : utils( std::move( space ), &backend.log() )
        {
        }
    };
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE( fixed_damping_convergence_strategy )

    fixed_damping_convergence_strategy( std::shared_ptr<Space> space, Log *log, const params &p = params() )
        : logged_obj_type( log, p ), parameters_( p ),
          space_( space ? std::move( space ) : throw std::invalid_argument( "Deflation requires a vector space" ) ),
          trial_( *space_ ), residual_( *space_ )
    {
        validate( parameters_ );
        for ( auto *buffer : { &trial_, &residual_ } )
        {
            buffer->init();
            buffer->start_use();
        }
        if ( parameters_.store_norms_history )
            history_.reserve( parameters_.max_iters_num );
    }

    fixed_damping_convergence_strategy( const utils_hierarchy &u, const params_hierarchy &p = params_hierarchy() )
        : fixed_damping_convergence_strategy( u.vec_space, u.log, p )
    {
    }

    void set_convergence_constants( T tolerance, int limit, T weight, bool history = false, bool verbose = true )
    {
        if ( limit < 1 )
            throw std::invalid_argument( "Deflation iteration limit must be positive" );
        params updated              = parameters_;
        updated.abs_tol             = tolerance;
        updated.max_iters_num       = static_cast<unsigned int>( limit );
        updated.weight              = weight;
        updated.store_norms_history = history;
        updated.verbose             = verbose;
        validate( updated );
        if ( updated.store_norms_history )
            history_.reserve( updated.max_iters_num );
        parameters_ = std::move( updated );
    }

    void reset_iterations()
    {
        iterations_   = 0;
        status_       = -1;
        increment_ok_ = true;
        history_.clear();
    }

    void increment_ok( bool ok )
    {
        increment_ok_ = ok;
    }
    int get_result_status() const
    {
        return status_;
    }
    unsigned int get_number_of_iterations() const
    {
        return iterations_;
    }
    std::vector<T> *get_norms_history_handle()
    {
        return &history_;
    }

    bool check_convergence(
        Operator *op, Projector *projector, Quality *quality, vector_type &state, vector_type &increment
    )
    {
        if ( status_ != -1 && status_ != 1 )
            return true;
        if ( status_ == -1 )
        {
            if ( !space_->is_valid_number( state ) )
                return finish( 11, quality, state );
            const T norm = physical_norm( *op, state );
            if ( !std::isfinite( norm ) )
                return finish( 2, quality, state );
            if ( norm > parameters_.maximum_residual )
                return finish( 12, quality, state );
            if ( norm <= parameters_.abs_tol )
                return finish( 0, quality, state );
            status_ = 1;
            return false;
        }
        if ( !increment_ok_ )
            return finish( 6, quality, state );
        if ( !space_->is_valid_number( increment ) )
            return finish( 7, quality, state );

        ++iterations_;
        space_->assign_lin_comb( T( 1 ), state, parameters_.weight, increment, *trial_ );
        if ( !space_->is_valid_number( *trial_ ) )
            return finish( 3, quality, state );
        if ( projector )
        {
            projector->apply( *trial_ );
            if ( !space_->is_valid_number( *trial_ ) )
                return finish( 10, quality, state );
        }
        const T norm = physical_norm( *op, *trial_ );
        if ( !std::isfinite( norm ) )
            return finish( 3, quality, state );
        if ( norm > parameters_.maximum_residual )
            return finish( 12, quality, state );

        // Accept finite physical residual growth: deflation must escape known roots.
        space_->assign( *trial_, state );
        if ( parameters_.store_norms_history )
            history_.push_back( norm );
        if ( parameters_.verbose )
            logged_obj_type::info_f(
                "iteration %u, physical residual %le, weight %le", iterations_, double( norm ),
                double( parameters_.weight )
            );
        if ( norm <= parameters_.abs_tol )
            return finish( 0, quality, state );
        if ( iterations_ >= parameters_.max_iters_num )
            return finish( 9, quality, state );
        return false;
    }

private:
    static void validate( const params &p )
    {
        if ( !std::isfinite( p.abs_tol ) || p.abs_tol < T( 0 ) || !std::isfinite( p.maximum_residual ) ||
             p.maximum_residual <= T( 0 ) || !std::isfinite( p.weight ) || p.weight <= T( 0 ) || p.weight > T( 1 ) ||
             p.max_iters_num == 0 || p.max_iters_num > static_cast<unsigned int>( std::numeric_limits<int>::max() ) )
            throw std::invalid_argument( "Invalid fixed-damping deflation parameters" );
    }

    T physical_norm( Operator &op, const vector_type &state )
    {
        const T checked_norm = op.original_residual_norm( state, *residual_ );
        return std::isfinite( checked_norm ) ? space_->norm( *residual_ ) : checked_norm;
    }

    bool finish( int status, Quality *quality, vector_type &state )
    {
        status_ = status;
        if ( parameters_.verbose )
            logged_obj_type::info_f( "status %d after %u corrections", status_, iterations_ );
        if ( quality && status != 11 )
            quality->calc( state );
        return true;
    }

    params                                         parameters_;
    std::shared_ptr<Space>                         space_;
    nmfd::detail::vector_wrap<Space, false, false> trial_, residual_;
    std::vector<T>                                 history_;
    unsigned int                                   iterations_   = 0;
    int                                            status_       = -1;
    bool                                           increment_ok_ = true;
};

}

#endif
