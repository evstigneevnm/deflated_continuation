#ifndef __SYMMETRY_FINITE_ISOTROPY_GEOMETRY_H__
#define __SYMMETRY_FINITE_ISOTROPY_GEOMETRY_H__

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/stabilizer_description.h>

namespace symmetry
{

// Actions are a complete, identity-first finite group with a supplied composition table.
template <class VectorOperations>
class finite_isotropy_geometry
{
public:
    using vector_type   = typename VectorOperations::vector_type;
    using scalar_type   = typename VectorOperations::scalar_type;
    using norm_type     = typename VectorOperations::norm_type;
    using registry_type = finite_action_registry<VectorOperations>;

    struct classification
    {
        std::uint64_t members = 0;
        norm_type     norm    = norm_type( 0 );
    };

    finite_isotropy_geometry(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<registry_type> actions,
        std::vector<std::size_t> composition, std::string fingerprint
    )
        : operations_( std::move( operations ) ), actions_( std::move( actions ) ),
          composition_( std::move( composition ) ), fingerprint_( std::move( fingerprint ) ),
          work_( checked_operations() ), average_( checked_operations() )
    {
        if ( !actions_ || actions_->vector_operations() != operations_.get() || fingerprint_.empty() )
            throw std::invalid_argument( "Finite isotropy requires matching actions and a group fingerprint" );
        count_    = actions_->size();
        revision_ = actions_->revision();
        if ( count_ == 0 || count_ > 64 || composition_.size() != count_ * count_ )
            throw std::invalid_argument( "Finite isotropy supports a complete group of at most 64 actions" );
        for ( std::size_t a = 0; a < count_; ++a )
        {
            bool inverse = false;
            for ( std::size_t b = 0; b < count_; ++b )
            {
                if ( product( a, b ) >= count_ )
                    throw std::invalid_argument( "Invalid finite group composition table" );
                inverse = inverse || ( product( a, b ) == 0 && product( b, a ) == 0 );
                for ( std::size_t c = 0; c < count_; ++c )
                    if ( product( b, c ) >= count_ || product( product( a, b ), c ) != product( a, product( b, c ) ) )
                        throw std::invalid_argument( "Finite group table must be associative" );
            }
            if ( product( 0, a ) != a || product( a, 0 ) != a || !inverse )
                throw std::invalid_argument( "Finite group table must have identity zero and inverses" );
        }
    }

    classification classify( const vector_type &state, norm_type tolerance )
    {
        check_registry();
        if ( !std::isfinite( tolerance ) || tolerance <= norm_type( 0 ) )
            throw std::invalid_argument( "Finite isotropy tolerance must be finite and positive" );
        if ( !operations_->is_valid_number( state ) )
            throw std::invalid_argument( "Finite isotropy requires a finite state" );
        classification result;
        result.norm = operations_->norm( state );
        if ( !std::isfinite( result.norm ) )
            throw std::overflow_error( "Finite isotropy state norm is not finite" );
        prepare();
        for ( std::size_t g = 0; g < count_; ++g )
        {
            actions_->apply( g, state, *work_ );
            operations_->add_mul( scalar_type( -1 ), state, *work_ );
            const auto defect = operations_->norm( *work_ );
            if ( !std::isfinite( defect ) )
                throw std::overflow_error( "Finite isotropy action produced a nonfinite defect" );
            if ( defect <= tolerance * result.norm )
                result.members |= std::uint64_t( 1 ) << g;
        }
        if ( ( result.members & 1 ) == 0 )
            throw std::runtime_error( "Declared identity does not fix the state" );
        for ( std::size_t a = 0; a < count_; ++a )
            for ( std::size_t b = 0; b < count_; ++b )
                if ( member( result, a ) && member( result, b ) && !member( result, product( a, b ) ) )
                    throw std::runtime_error(
                        "Approximate finite stabilizer is not closed at the requested tolerance"
                    );
        return result;
    }

    bool is_zero( const classification &value ) const
    {
        return value.norm == norm_type( 0 );
    }

    bool proper_stabilizer_increase( const classification &before, const classification &after ) const
    {
        return before.members != after.members && ( before.members & after.members ) == before.members;
    }

    stabilizer_description describe( const classification &value ) const
    {
        stabilizer_description result;
        result.representation    = "finite_actions";
        result.group_fingerprint = fingerprint_;
        for ( std::size_t g = 0; g < count_; ++g )
            if ( member( value, g ) )
                result.finite_elements.push_back( g );
        return result;
    }

    translation::orbit_type orbit_type( const classification &value ) const
    {
        translation::orbit_type result;
        result.finite_invariants = { static_cast<std::uint64_t>( describe( value ).finite_elements.size() ) };
        return result;
    }

    norm_type transverse_ratio( const vector_type &state, const classification &target )
    {
        check_registry();
        prepare();
        operations_->assign_scalar( scalar_type( 0 ), *average_ );
        const auto order = describe( target ).finite_elements.size();
        for ( std::size_t g = 0; g < count_; ++g )
            if ( member( target, g ) )
            {
                actions_->apply( g, state, *work_ );
                operations_->add_mul( scalar_type( 1 ) / scalar_type( order ), *work_, *average_ );
            }
        operations_->add_mul( scalar_type( -1 ), state, *average_ );
        const auto norm = operations_->norm( state );
        return norm == norm_type( 0 ) ? norm_type( 0 ) : operations_->norm( *average_ ) / norm;
    }

private:
    VectorOperations &checked_operations()
    {
        if ( !operations_ )
            throw std::invalid_argument( "Finite isotropy requires vector operations" );
        return *operations_;
    }
    std::size_t product( std::size_t a, std::size_t b ) const
    {
        return composition_[a * count_ + b];
    }
    static bool member( const classification &value, std::size_t g )
    {
        return ( value.members & ( std::uint64_t( 1 ) << g ) ) != 0;
    }
    void check_registry() const
    {
        if ( actions_->revision() != revision_ )
            throw std::logic_error( "Rebind finite isotropy geometry after changing its action registry" );
    }
    void prepare()
    {
        work_.init();
        work_.start_use();
        average_.init();
        average_.start_use();
    }

    std::shared_ptr<VectorOperations>                         operations_;
    std::shared_ptr<registry_type>                            actions_;
    std::vector<std::size_t>                                  composition_;
    std::string                                               fingerprint_;
    std::size_t                                               count_ = 0, revision_ = 0;
    nmfd::detail::vector_wrap<VectorOperations, false, false> work_, average_;
};

}

#endif
