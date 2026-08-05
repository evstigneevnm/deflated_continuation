#ifndef __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_ORBIT_ALIGNER_2D_H__
#define __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_ORBIT_ALIGNER_2D_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <symmetry/fourier/mode_descriptor.h>
#include <symmetry/fourier/residual_translation_group_2d.h>

namespace symmetry
{
namespace fourier
{

template<class VectorOperations>
class residual_translation_orbit_aligner_2d
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename vector_operations_type::scalar_type;
    using norm_type = typename vector_operations_type::norm_type;
    using vector_type = typename vector_operations_type::vector_type;
    using mode_type = mode_index<2>;
    using policy_type = residual_translation_policy_2d<scalar_type>;

    residual_translation_orbit_aligner_2d(
        vector_operations_type* vec_ops_,
        std::vector<mode_type> modes_,
        policy_type policy_ = {}):
        vec_ops(vec_ops_),
        modes(std::move(modes_)),
        policy(policy_)
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument(
                "residual translation aligner got null vector operations");
        }
        if(modes.size() != vec_ops->get_default_size())
        {
            throw std::invalid_argument(
                "residual translation mode layout has the wrong size");
        }
    }

    void align_orbit_closest_to_reference(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination)
    {
        require_vector_size(reference);
        require_vector_size(source);
        require_vector_size(destination);
        read_host(reference, reference_host);
        read_host(source, source_host);
        const auto group = build_residual_translation_group_2d(
            modes,
            source_host,
            policy);

        bool have_best = false;
        norm_type best_distance_squared = norm_type(0);
        best_host.resize(source_host.size());
        candidate_host.resize(source_host.size());
        for(const auto& character: group.characters)
        {
            norm_type distance_squared = norm_type(0);
            for(std::size_t index = 0; index < source_host.size(); ++index)
            {
                scalar_type value = source_host[index];
                if(character.compatible(modes[index]))
                {
                    value *= static_cast<scalar_type>(
                        character.sign(modes[index]));
                }
                candidate_host[index] = value;
                const norm_type difference = static_cast<norm_type>(
                    value - reference_host[index]);
                distance_squared += difference*difference;
            }
            if(!have_best || distance_squared < best_distance_squared)
            {
                have_best = true;
                best_distance_squared = distance_squared;
                best_host = candidate_host;
                last_character = character;
                last_character_valid = true;
            }
        }
        if(!have_best)
        {
            throw std::logic_error(
                "residual translation aligner found no representative");
        }
        vec_ops->set(best_host.data(), destination, best_host.size());
    }

    void apply_last_alignment(
        const vector_type& source,
        vector_type& destination)
    {
        require_vector_size(source);
        require_vector_size(destination);
        if(!last_character_valid)
        {
            vec_ops->assign(source, destination);
            return;
        }
        read_host(source, source_host);
        candidate_host.resize(source_host.size());
        for(std::size_t index = 0; index < source_host.size(); ++index)
        {
            candidate_host[index] = source_host[index];
            if(last_character.compatible(modes[index]))
            {
                candidate_host[index] *= static_cast<scalar_type>(
                    last_character.sign(modes[index]));
            }
        }
        vec_ops->set(
            candidate_host.data(),
            destination,
            candidate_host.size());
    }

    std::string orbit_definition_fingerprint() const
    {
        return "fourier_residual_translation_lattice_2d_v1";
    }

private:
    void require_vector_size(const vector_type& vector) const
    {
        if(vec_ops->get_size(vector) != modes.size())
        {
            throw std::invalid_argument(
                "residual translation aligner vector has the wrong size");
        }
    }

    void read_host(
        const vector_type& source,
        std::vector<scalar_type>& destination) const
    {
        destination.resize(modes.size());
        vec_ops->get(source, destination.data(), destination.size());
    }

    vector_operations_type* vec_ops;
    std::vector<mode_type> modes;
    policy_type policy;
    std::vector<scalar_type> reference_host;
    std::vector<scalar_type> source_host;
    std::vector<scalar_type> candidate_host;
    std::vector<scalar_type> best_host;
    residual_translation_character_2d last_character;
    bool last_character_valid = false;
};

} // namespace fourier
} // namespace symmetry

#endif
