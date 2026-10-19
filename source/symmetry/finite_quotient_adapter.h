#ifndef __SYMMETRY_FINITE_QUOTIENT_ADAPTER_H__
#define __SYMMETRY_FINITE_QUOTIENT_ADAPTER_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <nmfd/detail/vector_wrap.h>

#include <symmetry/finite_action_registry.h>

namespace symmetry
{

namespace detail
{

template <class Aligner, class VectorOperations, class Vector>
auto apply_orbit_alignment_to_tangent(
    Aligner *aligner, VectorOperations *, const Vector &source, Vector &destination, int
) -> decltype( aligner->stabilizer_differential_from_last( source, destination ), void() )
{
    aligner->stabilizer_differential_from_last( source, destination );
}

template <class Aligner, class VectorOperations, class Vector>
auto apply_orbit_alignment_to_tangent(
    Aligner *aligner, VectorOperations *, const Vector &source, Vector &destination, long
) -> decltype( aligner->apply_last_alignment( source, destination ), void() )
{
    aligner->apply_last_alignment( source, destination );
}

template <class Aligner, class VectorOperations, class Vector>
void apply_orbit_alignment_to_tangent(
    Aligner *, VectorOperations *vector_operations, const Vector &source, Vector &destination, ...
)
{
    vector_operations->assign( source, destination );
}

template <class OrbitAligner>
auto orbit_definition_fingerprint( const OrbitAligner *aligner, int )
    -> decltype( aligner->orbit_definition_fingerprint(), std::string() )
{
    return aligner->orbit_definition_fingerprint();
}

template <class OrbitAligner>
std::string orbit_definition_fingerprint( const OrbitAligner *, long )
{
    return {};
}

} // namespace detail

template <class VectorOperations, class ContinuousAdapter, class OrbitAligner = ContinuousAdapter>
class finite_quotient_adapter
{
public:
    using vector_operations_type      = VectorOperations;
    using continuous_adapter_type     = ContinuousAdapter;
    using orbit_aligner_type          = OrbitAligner;
    using finite_action_registry_type = finite_action_registry<VectorOperations>;
    using scalar_type                 = typename VectorOperations::scalar_type;
    using norm_type                   = typename VectorOperations::norm_type;
    using vector_type                 = typename VectorOperations::vector_type;

    template <
        class Aligner                                                                             = orbit_aligner_type,
        typename std::enable_if<std::is_same<Aligner, continuous_adapter_type>::value, int>::type = 0>
    finite_quotient_adapter(
        VectorOperations *vec_ops_, ContinuousAdapter *continuous_adapter_, finite_action_registry_type *finite_actions_
    )
        : finite_quotient_adapter( vec_ops_, continuous_adapter_, finite_actions_, continuous_adapter_ )
    {
    }

    finite_quotient_adapter(
        VectorOperations *vec_ops_, ContinuousAdapter *continuous_adapter_,
        finite_action_registry_type *finite_actions_, orbit_aligner_type *orbit_aligner_
    )
        : vec_ops(
              vec_ops_ ? vec_ops_ : throw std::invalid_argument( "finite_quotient_adapter got null vector operations" )
          ),
          continuous_adapter( continuous_adapter_ ), finite_actions( finite_actions_ ), orbit_aligner( orbit_aligner_ ),
          action_source_storage_( *vec_ops ), candidate_storage_( *vec_ops ), best_storage_( *vec_ops ),
          gradient_storage_( *vec_ops ), action_tangent_storage_( *vec_ops ), candidate_tangent_storage_( *vec_ops ),
          best_tangent_storage_( *vec_ops ), distance_storage_( *vec_ops )
    {
        if ( continuous_adapter == nullptr )
        {
            throw std::invalid_argument( "finite_quotient_adapter got null continuous adapter" );
        }
        if ( finite_actions == nullptr )
        {
            throw std::invalid_argument( "finite_quotient_adapter got null finite action registry" );
        }
        if ( orbit_aligner == nullptr )
        {
            throw std::invalid_argument( "finite_quotient_adapter got null orbit aligner" );
        }
        if ( finite_actions->size() == 0 )
        {
            throw std::invalid_argument( "finite_quotient_adapter needs at least one finite action" );
        }

        for ( auto *storage :
              { &action_source_storage_, &candidate_storage_, &best_storage_, &gradient_storage_,
                &action_tangent_storage_, &candidate_tangent_storage_, &best_tangent_storage_, &distance_storage_ } )
        {
            storage->init();
            storage->start_use();
        }
    }

    template <
        class Aligner                                                                = orbit_aligner_type,
        std::enable_if_t<std::is_same<Aligner, continuous_adapter_type>::value, int> = 0>
    finite_quotient_adapter(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<ContinuousAdapter> continuous,
        std::shared_ptr<finite_action_registry_type> actions
    )
        : finite_quotient_adapter( operations, continuous, std::move( actions ), continuous )
    {
    }

    finite_quotient_adapter(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<ContinuousAdapter> continuous,
        std::shared_ptr<finite_action_registry_type> actions, std::shared_ptr<orbit_aligner_type> aligner
    )
        : finite_quotient_adapter( operations.get(), continuous.get(), actions.get(), aligner.get() )
    {
        operations_owner_ = std::move( operations );
        continuous_owner_ = std::move( continuous );
        actions_owner_    = std::move( actions );
        aligner_owner_    = std::move( aligner );
    }

    finite_quotient_adapter( const finite_quotient_adapter & )            = delete;
    finite_quotient_adapter &operator=( const finite_quotient_adapter & ) = delete;

    ContinuousAdapter *continuous() const
    {
        return continuous_adapter;
    }

    finite_action_registry_type *finite_registry() const
    {
        return finite_actions;
    }

    std::string symmetry_definition_fingerprint() const
    {
        if ( !finite_actions->has_explicit_definition_fingerprint() )
        {
            return {};
        }
        std::string       fingerprint       = finite_actions->definition_fingerprint();
        const std::string orbit_fingerprint = detail::orbit_definition_fingerprint( orbit_aligner, 0 );
        if ( !orbit_fingerprint.empty() )
        {
            fingerprint += ":" + orbit_fingerprint;
        }
        return fingerprint;
    }

    std::vector<std::string> symmetry_action_names() const
    {
        if ( !finite_actions->has_explicit_definition_fingerprint() )
        {
            return {};
        }
        return finite_actions->action_names();
    }

    std::size_t last_action_index() const
    {
        return last_action_index_;
    }

    const std::string &last_action_name() const
    {
        return finite_actions->name( last_action_index_ );
    }

    void stabilize( const vector_type &source, vector_type &destination )
    {
        stabilize_canonical( source, destination );
    }

    void stabilize_canonical( const vector_type &source, vector_type &destination )
    {
        const std::size_t selected_action = select_canonical_action( source, destination );
        finite_actions->apply( selected_action, source, action_source );
        continuous_adapter->stabilize_canonical( action_source, destination );
        last_action_index_ = selected_action;
    }

    void
    stabilize_closest_to_reference( const vector_type &reference, const vector_type &source, vector_type &destination )
    {
        bool        have_best     = false;
        norm_type   best_distance = norm_type{};
        std::size_t best_index    = 0;

        for ( std::size_t action_index = 0; action_index < finite_actions->size(); ++action_index )
        {
            finite_actions->apply( action_index, source, action_source );
            orbit_aligner->align_orbit_closest_to_reference( reference, action_source, candidate );
            vec_ops->assign_mul( scalar_type( 1 ), candidate, scalar_type( -1 ), reference, distance );
            const norm_type candidate_distance = vec_ops->norm_l2( distance );

            if ( !have_best || candidate_distance < best_distance )
            {
                have_best     = true;
                best_distance = candidate_distance;
                best_index    = action_index;
                vec_ops->assign( candidate, best );
            }
        }

        vec_ops->assign( best, destination );
        last_action_index_ = best_index;
    }

    void align_orbit_closest_to_reference(
        const vector_type &reference, const vector_type &source, vector_type &destination
    )
    {
        stabilize_closest_to_reference( reference, source, destination );
    }

    void align_orbit_and_tangent_closest_to_reference(
        const vector_type &reference, const vector_type &source, const vector_type &source_tangent,
        vector_type &destination, vector_type &tangent_destination
    )
    {
        bool        have_best     = false;
        norm_type   best_distance = norm_type{};
        std::size_t best_index    = 0;

        for ( std::size_t action_index = 0; action_index < finite_actions->size(); ++action_index )
        {
            finite_actions->apply( action_index, source, action_source );
            finite_actions->apply( action_index, source_tangent, action_tangent );
            orbit_aligner->align_orbit_closest_to_reference( reference, action_source, candidate );
            detail::apply_orbit_alignment_to_tangent( orbit_aligner, vec_ops, action_tangent, candidate_tangent, 0 );
            vec_ops->assign_mul( scalar_type( 1 ), candidate, scalar_type( -1 ), reference, distance );
            const norm_type candidate_distance = vec_ops->norm_l2( distance );
            if ( !have_best || candidate_distance < best_distance )
            {
                have_best     = true;
                best_distance = candidate_distance;
                best_index    = action_index;
                vec_ops->assign( candidate, best );
                vec_ops->assign( candidate_tangent, best_tangent );
            }
        }
        if ( !have_best )
        {
            throw std::logic_error( "finite quotient adapter found no endpoint representative" );
        }
        vec_ops->assign( best, destination );
        vec_ops->assign( best_tangent, tangent_destination );
        last_action_index_ = best_index;
    }

    void pullback_distance_gradient(
        const vector_type &source, const vector_type &slice_state, const vector_type &slice_gradient,
        vector_type &gradient
    )
    {
        pullback_canonical_distance_gradient( source, slice_state, slice_gradient, gradient );
    }

    void pullback_canonical_distance_gradient(
        const vector_type &source, const vector_type &slice_state, const vector_type &slice_gradient,
        vector_type &gradient
    )
    {
        const std::size_t selected_action = select_canonical_action( source, candidate );
        finite_actions->apply( selected_action, source, action_source );
        continuous_adapter->pullback_canonical_distance_gradient(
            action_source, slice_state, slice_gradient, action_gradient
        );
        finite_actions->pullback( selected_action, action_gradient, gradient );
        last_action_index_ = selected_action;
    }

private:
    std::shared_ptr<VectorOperations>            operations_owner_;
    std::shared_ptr<ContinuousAdapter>           continuous_owner_;
    std::shared_ptr<finite_action_registry_type> actions_owner_;
    std::shared_ptr<orbit_aligner_type>          aligner_owner_;
    std::size_t select_canonical_action( const vector_type &source, vector_type &destination )
    {
        bool                     have_best = false;
        std::vector<scalar_type> best_host;
        std::size_t              best_index = 0;
        const scalar_type        tolerance  = canonical_zero_tolerance( source );

        for ( std::size_t action_index = 0; action_index < finite_actions->size(); ++action_index )
        {
            finite_actions->apply( action_index, source, action_source );
            continuous_adapter->stabilize_canonical( action_source, candidate );
            auto candidate_host = get_host_state( candidate );
            zero_small_components( candidate_host, tolerance );

            if ( !have_best || lexicographically_greater( candidate_host, best_host, tolerance ) )
            {
                have_best  = true;
                best_index = action_index;
                best_host  = candidate_host;
                vec_ops->assign( candidate, best );
            }
        }

        vec_ops->assign( best, destination );
        last_action_index_ = best_index;
        return best_index;
    }

    std::vector<scalar_type> get_host_state( const vector_type &x ) const
    {
        std::vector<scalar_type> host( vec_ops->get_size( x ), scalar_type( 0 ) );
        vec_ops->get( x, host.data(), host.size() );
        return host;
    }

    scalar_type canonical_zero_tolerance( const vector_type &source ) const
    {
        const auto  source_host = get_host_state( source );
        scalar_type scale       = scalar_type( 1 );
        for ( const auto &value : source_host )
        {
            scale = std::max( scale, static_cast<scalar_type>( std::abs( value ) ) );
        }
        return scalar_type( 4096 ) * std::numeric_limits<scalar_type>::epsilon() * scale;
    }

    static void zero_small_components( std::vector<scalar_type> &values, const scalar_type tolerance )
    {
        for ( auto &value : values )
        {
            if ( std::abs( value ) <= tolerance )
            {
                value = scalar_type( 0 );
            }
        }
    }

    static bool lexicographically_greater(
        const std::vector<scalar_type> &left, const std::vector<scalar_type> &right, const scalar_type tolerance
    )
    {
        const std::size_t n = std::min( left.size(), right.size() );
        for ( std::size_t i = 0; i < n; ++i )
        {
            const scalar_type delta = left[i] - right[i];
            if ( delta > tolerance )
            {
                return true;
            }
            if ( delta < -tolerance )
            {
                return false;
            }
        }
        return left.size() > right.size();
    }

private:
    VectorOperations                                         *vec_ops;
    ContinuousAdapter                                        *continuous_adapter;
    finite_action_registry_type                              *finite_actions;
    orbit_aligner_type                                       *orbit_aligner;
    nmfd::detail::vector_wrap<VectorOperations, false, false> action_source_storage_, candidate_storage_, best_storage_,
        gradient_storage_, action_tangent_storage_, candidate_tangent_storage_, best_tangent_storage_,
        distance_storage_;
    vector_type &action_source      = *action_source_storage_;
    vector_type &candidate          = *candidate_storage_;
    vector_type &best               = *best_storage_;
    vector_type &action_gradient    = *gradient_storage_;
    vector_type &action_tangent     = *action_tangent_storage_;
    vector_type &candidate_tangent  = *candidate_tangent_storage_;
    vector_type &best_tangent       = *best_tangent_storage_;
    vector_type &distance           = *distance_storage_;
    std::size_t  last_action_index_ = 0;
};

} // namespace symmetry

#endif
