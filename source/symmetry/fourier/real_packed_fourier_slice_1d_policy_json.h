#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_POLICY_JSON_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_POLICY_JSON_H__

#include <cstddef>
#include <stdexcept>
#include <string>

#include <contrib/json/nlohmann/json.hpp>
#include <symmetry/fourier/real_packed_fourier_slice_1d_policy.h>

namespace symmetry
{
namespace fourier
{

template<class T>
real_packed_fourier_slice_1d_policy<T> read_real_packed_fourier_slice_1d_policy_object(
    const nlohmann::json& config,
    real_packed_fourier_slice_1d_policy<T> policy,
    const std::string& context)
{
    const std::string type = config.value(
        "type",
        std::string(real_packed_fourier_1d_stabilizer_policy_name(policy.stabilizer)));
    if(type == "single_mode")
    {
        policy.stabilizer = real_packed_fourier_1d_stabilizer_policy::single_mode;
    }
    else if(type == "lsq_multimode")
    {
        policy.stabilizer = real_packed_fourier_1d_stabilizer_policy::lsq_multimode;
    }
    else
    {
        throw std::runtime_error("unknown " + context + ".type '" + type + "'");
    }

    policy.relative_active_mode_tolerance = config.value(
        "relative_active_mode_tolerance", policy.relative_active_mode_tolerance);
    policy.continuation_mode_switch_ratio = config.value(
        "continuation_mode_switch_ratio", policy.continuation_mode_switch_ratio);
    policy.tangent_continuity_weight = config.value(
        "tangent_continuity_weight", policy.tangent_continuity_weight);
    policy.tangent_backward_penalty = config.value(
        "tangent_backward_penalty", policy.tangent_backward_penalty);
    policy.lsq.mode_min = config.value("mode_min", policy.lsq.mode_min);
    policy.lsq.mode_max = config.value("mode_max", policy.lsq.mode_max);
    policy.lsq.max_active_modes = config.value("max_active_modes", policy.lsq.max_active_modes);
    policy.lsq.grid_points = config.value("grid_points", policy.lsq.grid_points);
    policy.lsq.newton_iterations = config.value("newton_iterations", policy.lsq.newton_iterations);
    policy.lsq.prefer_trivial_residual_group = config.value(
        "prefer_trivial_residual_group", policy.lsq.prefer_trivial_residual_group);
    policy.lsq.minimum_coprime_relative_score = config.value(
        "minimum_coprime_relative_score", policy.lsq.minimum_coprime_relative_score);
    policy.local_representative_relative_tolerance = config.value(
        "local_representative_relative_tolerance",
        policy.local_representative_relative_tolerance);
    policy.validate();
    return policy;
}

template<class T>
real_packed_fourier_slice_1d_policy<T> read_real_packed_fourier_slice_1d_policy(
    const nlohmann::json& root,
    real_packed_fourier_slice_1d_policy<T> policy = real_packed_fourier_slice_1d_policy<T>())
{
    const nlohmann::json* config = nullptr;
    if(root.contains("nonlinear_operator") &&
       root["nonlinear_operator"].contains("symmetry_stabilizer"))
    {
        config = &root["nonlinear_operator"]["symmetry_stabilizer"];
    }
    if(root.contains("symmetry_stabilizer"))
    {
        config = &root["symmetry_stabilizer"];
    }

    if(config == nullptr)
    {
        policy.validate();
        return policy;
    }

    return read_real_packed_fourier_slice_1d_policy_object<T>(
        *config, policy, "symmetry_stabilizer");
}

template<class T>
real_packed_fourier_slice_1d_policy<T> read_real_packed_fourier_slice_1d_policy(
    const nlohmann::json& root,
    const std::string& config_key,
    real_packed_fourier_slice_1d_policy<T> policy)
{
    if(!root.contains(config_key))
    {
        policy.validate();
        return policy;
    }
    return read_real_packed_fourier_slice_1d_policy_object<T>(
        root.at(config_key), policy, config_key);
}

} // namespace fourier
} // namespace symmetry

#endif
