#ifndef __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_ORBIT_ALIGNER_2D_H__
#define __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_ORBIT_ALIGNER_2D_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <symmetry/fourier/mode_descriptor.h>
#include <symmetry/fourier/residual_translation_group_2d.h>
#include <symmetry/fourier/residual_translation_workspace_2d.h>

namespace symmetry
{
namespace fourier
{

template <class VectorOperations>
class residual_translation_orbit_aligner_2d
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type            = typename vector_operations_type::scalar_type;
    using norm_type              = typename vector_operations_type::norm_type;
    using vector_type            = typename vector_operations_type::vector_type;
    using mode_type              = mode_index<2>;
    using policy_type            = residual_translation_policy_2d<scalar_type>;

    class prepared_source_type
    {
    public:
        std::size_t candidate_count() const
        {
            return candidates.size();
        }

    private:
        friend class residual_translation_orbit_aligner_2d;
        struct candidate_type
        {
            residual_translation_character_2d character;
            norm_type                         error_bound;
            std::size_t                       representative;
        };
        const residual_translation_orbit_aligner_2d *owner = nullptr;
        std::vector<candidate_type>                  candidates;
        std::vector<std::size_t>                     representatives;
        norm_type                                    within_group_bound = norm_type( 0 );
    };

    residual_translation_orbit_aligner_2d(
        vector_operations_type *vec_ops_, std::vector<mode_type> modes_, policy_type policy_ = {}
    )
        : vec_ops(
              vec_ops_ ? vec_ops_
                       : throw std::invalid_argument( "residual translation aligner got null vector operations" )
          ),
          modes( std::move( modes_ ) ), policy( policy_ ), workspace( *vec_ops, modes )
    {
        if ( modes.size() != vec_ops->get_default_size() )
        {
            throw std::invalid_argument( "residual translation mode layout has the wrong size" );
        }
        detail::validate_policy( policy );
        candidate_distances.reserve( policy.maximum_proposal_characters );
    }

    class reference_scope
    {
    public:
        reference_scope( residual_translation_orbit_aligner_2d &aligner, const vector_type &reference )
            : aligner_( aligner )
        {
            if ( aligner_.prepared_reference != nullptr )
                throw std::logic_error( "residual translation reference scope is already active" );
            aligner_.prepared_group     = aligner_.build_group( reference );
            aligner_.prepared_reference = &reference;
        }
        ~reference_scope()
        {
            aligner_.prepared_reference = nullptr;
        }
        reference_scope( const reference_scope & )            = delete;
        reference_scope &operator=( const reference_scope & ) = delete;

    private:
        residual_translation_orbit_aligner_2d &aligner_;
    };

    // Explicitly scoped reuse across finite actions; never cache by vector address between calls.
    reference_scope prepare_reference( const vector_type &reference )
    {
        return reference_scope( *this, reference );
    }

    residual_translation_orbit_aligner_2d(
        std::shared_ptr<vector_operations_type> operations, std::vector<mode_type> modes, policy_type policy = {}
    )
        : residual_translation_orbit_aligner_2d( operations.get(), std::move( modes ), policy )
    {
        operations_owner_ = std::move( operations );
    }

    void align_orbit_closest_to_reference(
        const vector_type &reference, const vector_type &source, vector_type &destination
    )
    {
        align( reference, source, destination, nullptr );
    }

    // Preparation belongs to an immutable source, never to a vector address.
    prepared_source_type prepare_source_orbit( const vector_type &source )
    {
        const auto      strict    = build_group( source );
        const norm_type tolerance = strict.active_tolerance;
        // Any coefficient above this bound must be compatible with an admissible character.
        const norm_type threshold = tolerance * std::sqrt( static_cast<norm_type>( modes.size() ) ) / norm_type( 2 );
        auto            proposal_policy           = policy;
        proposal_policy.absolute_active_tolerance = threshold;
        proposal_policy.relative_active_tolerance = norm_type( 0 );
        const auto &mask                          = workspace.active_mask( source, threshold );
        const auto  proposals =
            build_residual_translation_group_2d( modes, mask, strict.maximum_amplitude, proposal_policy, true );
        const auto tail_bound = workspace.discarded_error_bound( source, threshold );
        if ( proposals.active_rank < strict.active_rank && tail_bound > tolerance )
            throw std::runtime_error( "residual translation proposal support is unresolved" );
        auto characters = strict.characters;
        characters.insert( characters.end(), proposals.characters.begin(), proposals.characters.end() );
        auto unique = normalize_characters( std::move( characters ) );
        if ( unique.size() > policy.maximum_proposal_characters )
            throw std::runtime_error( "residual translation proposal budget exceeded" );
        prepared_source_type prepared;
        prepared.owner              = this;
        prepared.within_group_bound = tail_bound * std::sqrt( static_cast<norm_type>( modes.size() ) );
        std::vector<std::size_t> strong_modes;
        for ( std::size_t i = 0; i < modes.size(); ++i )
            if ( ( mask[i / 8] & ( 1u << ( i % 8 ) ) ) != 0 )
                strong_modes.push_back( i );
        for ( const auto &character : unique )
        {
            const auto defect = workspace.incompatible_norm_sq( character, source );
            if ( norm_type( 4 ) * defect <= tolerance * tolerance * static_cast<norm_type>( modes.size() ) )
            {
                std::size_t representative = prepared.candidates.size();
                for ( const auto index : prepared.representatives )
                {
                    const auto &other = prepared.candidates[index].character;
                    if ( std::all_of( strong_modes.begin(), strong_modes.end(), [&]( const auto i ) {
                             return detail::retained_translation_sign( character, modes[i] ) ==
                                    detail::retained_translation_sign( other, modes[i] );
                         } ) )
                    {
                        representative = index;
                        break;
                    }
                }
                if ( representative == prepared.candidates.size() )
                    prepared.representatives.push_back( representative );
                prepared.candidates.push_back(
                    { character,
                      modes.empty() ? norm_type( 0 )
                                    : norm_type( 2 ) * std::sqrt( defect / static_cast<norm_type>( modes.size() ) ),
                      representative }
                );
            }
        }
        if ( prepared.candidates.empty() )
            throw std::logic_error( "residual translation source has no identity representative" );
        return prepared;
    }

    // A fixed source orbit is queried without transforming the reference.
    void align_source_orbit_closest_to_reference(
        const vector_type &reference, const vector_type &source, vector_type &destination
    )
    {
        const auto prepared = prepare_source_orbit( source );
        align_source_orbit_closest_to_reference( reference, source, prepared, destination );
    }

    bool align_source_orbit_closest_to_reference(
        const vector_type &reference, const vector_type &source, const prepared_source_type &prepared,
        vector_type &destination, norm_type maximum_distance = std::numeric_limits<norm_type>::infinity()
    )
    {
        last_character_valid = false;
        require_vector_size( reference );
        require_vector_size( source );
        require_vector_size( destination );
        if ( prepared.owner != this || prepared.candidates.empty() )
            throw std::invalid_argument( "residual translation source preparation belongs to another aligner" );
        if ( std::isnan( maximum_distance ) || maximum_distance < norm_type( 0 ) )
            throw std::invalid_argument( "residual translation distance bound must be nonnegative" );
        workspace.maximum_amplitude( reference );
        const auto rounding =
            norm_type( 8 ) * static_cast<norm_type>( modes.size() + 1 ) * std::numeric_limits<norm_type>::epsilon();
        const auto  upper    = maximum_distance * ( norm_type( 1 ) + rounding );
        norm_type   best     = modes.empty() ? norm_type( 0 ) : upper * upper * static_cast<norm_type>( modes.size() );
        std::size_t selected = 0;
        candidate_distances.resize( prepared.candidates.size() );
        const auto evaluate = [&]( std::size_t index ) {
            const auto &candidate      = prepared.candidates[index];
            const auto  distance       = workspace.distance_squared( candidate.character, reference, source );
            candidate_distances[index] = distance;
            if ( distance < best || ( distance == best && ( !last_character_valid || index < selected ) ) )
            {
                best                 = distance;
                selected             = index;
                last_character       = candidate.character;
                last_character_valid = true;
            }
        };
        for ( const auto index : prepared.representatives )
            evaluate( index );
        for ( std::size_t i = 0; i < prepared.candidates.size(); ++i )
        {
            const auto representative = prepared.candidates[i].representative;
            if ( representative == i )
                continue;
            // Equal strong-mode actions differ only in the bounded tail.
            const auto group_distance = std::sqrt( candidate_distances[representative] );
            const auto roundoff =
                rounding < norm_type( .5 )
                    ? rounding / ( norm_type( 1 ) - rounding ) * std::max( group_distance, prepared.within_group_bound )
                    : std::numeric_limits<norm_type>::infinity();
            if ( !std::isfinite( group_distance ) || !std::isfinite( roundoff ) ||
                 group_distance - prepared.within_group_bound - roundoff <= std::sqrt( best ) )
                evaluate( i );
        }
        if ( !last_character_valid )
            return false;
        workspace.apply( last_character, source, destination );
        return true;
    }

    void align_orbit_and_tangent_closest_to_reference(
        const vector_type &reference, const vector_type &source, const vector_type &source_tangent,
        vector_type &destination, vector_type &tangent_destination
    )
    {
        require_vector_size( source_tangent );
        require_vector_size( tangent_destination );
        align( reference, source, destination, &source_tangent );
        workspace.apply( last_character, source_tangent, tangent_destination );
    }

    void apply_last_alignment( const vector_type &source, vector_type &destination )
    {
        require_vector_size( source );
        require_vector_size( destination );
        if ( !last_character_valid )
        {
            vec_ops->assign( source, destination );
            return;
        }
        const auto tolerance = active_tolerance( workspace.maximum_amplitude( source ) );
        if ( !workspace.preserves_subspace( last_character, source, tolerance ) )
        {
            throw std::domain_error( "Residual translation does not preserve the tangent subspace" );
        }
        workspace.apply( last_character, source, destination );
    }

    std::string orbit_definition_fingerprint() const
    {
        return "fourier_residual_translation_lattice_2d_v2";
    }

private:
    using character_type = residual_translation_character_2d;

    void align(
        const vector_type &reference, const vector_type &source, vector_type &destination, const vector_type *tangent
    )
    {
        last_character_valid = false;
        require_vector_size( reference );
        require_vector_size( source );
        require_vector_size( destination );
        const auto source_group    = build_group( source );
        const auto reference_group = prepared_reference == &reference ? prepared_group : build_group( reference );
        auto       characters      = reference_group.characters;
        characters.insert( characters.end(), source_group.characters.begin(), source_group.characters.end() );
        const norm_type tolerance = std::max( reference_group.active_tolerance, source_group.active_tolerance );
        const norm_type tangent_tolerance =
            tangent ? active_tolerance( workspace.maximum_amplitude( *tangent ) ) : norm_type( 0 );

        const auto unique = normalize_characters( std::move( characters ) );

        bool      have_best             = false;
        norm_type best_distance_squared = norm_type( 0 );
        for ( const auto &character : unique )
        {
            if ( !workspace.preserves_subspace( character, reference, tolerance ) )
                continue;
            if ( !workspace.preserves_subspace( character, source, tolerance ) ||
                 ( tangent && !workspace.preserves_subspace( character, *tangent, tangent_tolerance ) ) )
            {
                continue;
            }
            const norm_type distance_squared = workspace.distance_squared( character, reference, source );
            if ( !have_best || distance_squared < best_distance_squared )
            {
                have_best             = true;
                best_distance_squared = distance_squared;
                last_character        = character;
                last_character_valid  = true;
            }
        }
        if ( !have_best )
        {
            throw std::logic_error( "residual translation aligner found no representative" );
        }
        workspace.apply( last_character, source, destination );
    }

    static std::vector<character_type> normalize_characters( std::vector<character_type> characters )
    {
        std::sort( characters.begin(), characters.end(), []( const auto &left, const auto &right ) {
            return std::tie( left.denominator, left.numerator_x, left.numerator_y ) <
                   std::tie( right.denominator, right.numerator_x, right.numerator_y );
        } );
        std::vector<character_type> unique;
        for ( auto character : characters )
        {
            const auto divisor =
                std::gcd( character.denominator, std::gcd( character.numerator_x, character.numerator_y ) );
            character.denominator /= divisor;
            character.numerator_x /= divisor;
            character.numerator_y /= divisor;
            if ( std::none_of( unique.begin(), unique.end(), [&]( const auto &other ) {
                     return std::tie( character.denominator, character.numerator_x, character.numerator_y ) ==
                            std::tie( other.denominator, other.numerator_x, other.numerator_y );
                 } ) )
                unique.push_back( character );
        }
        return unique;
    }

    norm_type active_tolerance( norm_type maximum ) const
    {
        return std::max( policy.absolute_active_tolerance, policy.relative_active_tolerance * maximum );
    }

    residual_translation_group_result_2d<scalar_type> build_group( const vector_type &source )
    {
        require_vector_size( source );
        const norm_type maximum = workspace.maximum_amplitude( source );
        return build_residual_translation_group_2d(
            modes, workspace.active_mask( source, active_tolerance( maximum ) ), maximum, policy
        );
    }

    void require_vector_size( const vector_type &vector ) const
    {
        if ( vec_ops->get_size( vector ) != modes.size() )
        {
            throw std::invalid_argument( "residual translation aligner vector has the wrong size" );
        }
    }

    std::shared_ptr<vector_operations_type>                           operations_owner_;
    vector_operations_type                                           *vec_ops;
    std::vector<mode_type>                                            modes;
    policy_type                                                       policy;
    detail::residual_translation_workspace_2d<vector_operations_type> workspace;
    const vector_type                                                *prepared_reference = nullptr;
    residual_translation_group_result_2d<scalar_type>                 prepared_group;
    residual_translation_character_2d                                 last_character;
    std::vector<norm_type>                                            candidate_distances;
    bool                                                              last_character_valid = false;
};

} // namespace fourier
} // namespace symmetry

#endif
