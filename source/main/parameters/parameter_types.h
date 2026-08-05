#ifndef __MAIN_PARAMETER_TYPES_H__
#define __MAIN_PARAMETER_TYPES_H__

/**
*
* Main structure to hold all parameters of execution.
* It is to be filled by the json config file from each project.
*
*/

#include <vector>
#include <string>
#include <iostream>
#include <cstddef>
#include <cstdint>

#include <stability/analysis/matrix_free_stability_config.h>

namespace main_classes
{

template <typename T>
struct parameters
{

    struct deflation_continuation_s
    {
        struct linear_solver_extended_s
        {
            unsigned int lin_solver_max_it;
            unsigned int use_precond_resid;
            unsigned int resid_recalc_freq;
            unsigned int basis_size;
            T    lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
            bool is_small_alpha;
            bool save_convergence_history;
            bool divide_out_norms_by_rel_base;
            bool verbose;


            void set_default()
            {
                lin_solver_max_it            = 1500;
                use_precond_resid            = 1;
                resid_recalc_freq            = 1;
                basis_size                   = 4;
                lin_solver_tol               = 5.0e-3;
                is_small_alpha               = false;
                save_convergence_history     = true;
                divide_out_norms_by_rel_base = true;
                verbose                      = true;
            }

            void plot_all()
            {
                std::cout << "||  |==lin_solver_max_it: " << lin_solver_max_it << std::endl;
                std::cout << "||  |==use_precond_resid: " << use_precond_resid << std::endl;
                std::cout << "||  |==resid_recalc_freq: " << resid_recalc_freq << std::endl;
                std::cout << "||  |==basis_size: " << basis_size << std::endl;
                std::cout << "||  |==lin_solver_tol: " << lin_solver_tol << std::endl;
                std::cout << "||  |==is_small_alpha: " << is_small_alpha << std::endl;
                std::cout << "||  |==save_convergence_history: " << save_convergence_history << std::endl;
                std::cout << "||  |==divide_out_norms_by_rel_base: " << divide_out_norms_by_rel_base << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };
        struct newton_extended_continuation_s
        {
            unsigned int newton_max_it;
            T            newton_wight;
            bool         store_norms_history;
            bool         verbose;
            T            tolerance;
            T            relax_tolerance_factor;
            int          relax_tolerance_steps;
            unsigned int stagnation_max;
            T            maximum_norm_increase;
            T            newton_wight_threshold;
            void         set_default()
            {
                newton_max_it          = 300;
                newton_wight           = T( 1.0 );
                store_norms_history    = true;
                verbose                = true;
                tolerance              = 1.0e-9;
                relax_tolerance_factor = 10;
                relax_tolerance_steps  = 1;
                stagnation_max         = 10;
                maximum_norm_increase  = 2.0;
                newton_wight_threshold = 1.0e-12;
            }
            void plot_all()
            {
                std::cout << "||  |==newton_max_it: " << newton_max_it << std::endl;
                std::cout << "||  |==newton_wight: " << newton_wight << std::endl;
                std::cout << "||  |==store_norms_history: " << store_norms_history << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
                std::cout << "||  |==tolerance: " << tolerance << std::endl;
                std::cout << "||  |==relax_tolerance_factor: " << relax_tolerance_factor << std::endl;
                std::cout << "||  |==relax_tolerance_steps: " << relax_tolerance_steps << std::endl;
                std::cout << "||  |==stagnation_max: " << stagnation_max << std::endl;
                std::cout << "||  |==maximum_norm_increase: " << maximum_norm_increase << std::endl;
                std::cout << "||  |==newton_wight_threshold: " << newton_wight_threshold << std::endl;
            }
        };
        struct newton_extended_deflation_s
        {
            unsigned int newton_max_it;
            T            newton_wight;
            bool         store_norms_history;
            bool         verbose;
            T            tolerance;
            void         set_default()
            {
                newton_max_it       = 300;
                newton_wight        = T( 0.5 );
                store_norms_history = true;
                verbose             = true;
                tolerance           = 1.0e-9;
            }
            void plot_all()
            {
                std::cout << "||  |==newton_max_it: " << newton_max_it << std::endl;
                std::cout << "||  |==newton_wight: " << newton_wight << std::endl;
                std::cout << "||  |==store_norms_history: " << store_norms_history << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
                std::cout << "||  |==tolerance: " << tolerance << std::endl;
            }
        };

        struct restart_policy_s
        {
            struct knot_relocation_s
            {
                struct manual_override_s
                {
                    T requested;
                    T effective;
                    std::string reason;
                };

                bool enabled;
                std::string registry_file;
                T min_shift_abs;
                T max_shift_abs;
                unsigned int candidate_count;
                bool prefer_positive_shift;
                bool require_all_intersections;
                bool save_registry;
                std::vector<manual_override_s> manual_overrides;

                void set_default()
                {
                    enabled = false;
                    registry_file = "knot_registry.json";
                    min_shift_abs = T(1.0e-6);
                    max_shift_abs = T(5.0e-2);
                    candidate_count = 12;
                    prefer_positive_shift = true;
                    require_all_intersections = true;
                    save_registry = true;
                    manual_overrides.clear();
                }

                void plot_all()
                {
                    std::cout << "||  |  |==enabled: " << enabled << std::endl;
                    std::cout << "||  |  |==registry_file: " << registry_file << std::endl;
                    std::cout << "||  |  |==min_shift_abs: " << min_shift_abs << std::endl;
                    std::cout << "||  |  |==max_shift_abs: " << max_shift_abs << std::endl;
                    std::cout << "||  |  |==candidate_count: " << candidate_count << std::endl;
                    std::cout << "||  |  |==prefer_positive_shift: " << prefer_positive_shift << std::endl;
                    std::cout << "||  |  |==require_all_intersections: " << require_all_intersections << std::endl;
                    std::cout << "||  |  |==save_registry: " << save_registry << std::endl;
                    std::cout << "||  |  |==manual_overrides: ";
                    for(const auto& item: manual_overrides)
                    {
                        std::cout
                            << item.requested << "->"
                            << item.effective << " ";
                    }
                    std::cout << std::endl;
                }
            };

            struct seed_schedule_s
            {
                bool enabled;
                std::string registry_file;
                bool save_registry;

                void set_default()
                {
                    enabled = false;
                    registry_file =
                        "deflation_seed_registry.json";
                    save_registry = true;
                }

                void plot_all()
                {
                    std::cout << "||  |  |==enabled: "
                              << enabled << std::endl;
                    std::cout << "||  |  |==registry_file: "
                              << registry_file << std::endl;
                    std::cout << "||  |  |==save_registry: "
                              << save_registry << std::endl;
                }
            };

            struct failed_candidate_registry_s
            {
                bool enabled;
                std::string registry_file;
                std::string state_directory;
                std::uint64_t policy_generation;

                void set_default()
                {
                    enabled = true;
                    registry_file =
                        "failed_continuations/registry.json";
                    state_directory =
                        "failed_continuations/states";
                    policy_generation = 1;
                }

                void plot_all()
                {
                    std::cout << "||  |  |==enabled: " << enabled << std::endl;
                    std::cout << "||  |  |==registry_file: "
                              << registry_file << std::endl;
                    std::cout << "||  |  |==state_directory: "
                              << state_directory << std::endl;
                    std::cout << "||  |  |==policy_generation: "
                              << policy_generation << std::endl;
                }
            };

            struct recovery_registry_s
            {
                bool enabled;
                std::string registry_file;
                std::string checkpoint_directory;
                std::uint64_t policy_generation;
                bool process_before_deflation;

                void set_default()
                {
                    enabled = true;
                    registry_file =
                        "continuation_recovery_registry.json";
                    checkpoint_directory = "recovery_checkpoints";
                    policy_generation = 1;
                    process_before_deflation = false;
                }

                void plot_all()
                {
                    std::cout << "||  |  |==enabled: " << enabled << std::endl;
                    std::cout << "||  |  |==registry_file: "
                              << registry_file << std::endl;
                    std::cout << "||  |  |==checkpoint_directory: "
                              << checkpoint_directory << std::endl;
                    std::cout << "||  |  |==policy_generation: "
                              << policy_generation << std::endl;
                    std::cout << "||  |  |==process_before_deflation: "
                              << process_before_deflation << std::endl;
                }
            };

            struct topology_registry_s
            {
                bool enabled;
                std::string registry_file;
                std::string endpoint_directory;
                std::uint64_t policy_generation;
                T absolute_parameter_tolerance;
                T relative_parameter_tolerance;
                T state_tolerance;
                T minimum_tangent_line_similarity;
                bool record_transverse_junctions;

                void set_default()
                {
                    enabled = true;
                    registry_file = "branch_topology.json";
                    endpoint_directory = "branch_topology/endpoints";
                    policy_generation = 1;
                    absolute_parameter_tolerance = T(1.0e-7);
                    relative_parameter_tolerance = T(1.0e-9);
                    state_tolerance = T(1.0e-8);
                    minimum_tangent_line_similarity = T(0.9);
                    record_transverse_junctions = true;
                }

                void plot_all()
                {
                    std::cout << "||  |  |==enabled: " << enabled << std::endl;
                    std::cout << "||  |  |==registry_file: "
                              << registry_file << std::endl;
                    std::cout << "||  |  |==endpoint_directory: "
                              << endpoint_directory << std::endl;
                    std::cout << "||  |  |==policy_generation: "
                              << policy_generation << std::endl;
                    std::cout << "||  |  |==absolute_parameter_tolerance: "
                              << absolute_parameter_tolerance << std::endl;
                    std::cout << "||  |  |==relative_parameter_tolerance: "
                              << relative_parameter_tolerance << std::endl;
                    std::cout << "||  |  |==state_tolerance: "
                              << state_tolerance << std::endl;
                    std::cout << "||  |  |==minimum_tangent_line_similarity: "
                              << minimum_tangent_line_similarity << std::endl;
                    std::cout << "||  |  |==record_transverse_junctions: "
                              << record_transverse_junctions << std::endl;
                }
            };

            bool allow_incomplete_restart_intersections;
            bool allow_knot_interpolation_failure;
            bool allow_failed_continuation_curve_save;
            bool preserve_partial_curves;
            bool check_duplicate_after_deflation;
            unsigned int duplicate_after_deflation_retries;
            T duplicate_after_deflation_tolerance;
            unsigned int max_failed_continuations_per_knot;
            T failed_continuation_rejection_tolerance;
            knot_relocation_s knot_relocation;
            seed_schedule_s seed_schedule;
            failed_candidate_registry_s failed_candidate_registry;
            recovery_registry_s recovery_registry;
            topology_registry_s topology_registry;

            void set_default()
            {
                allow_incomplete_restart_intersections = false;
                allow_knot_interpolation_failure = false;
                allow_failed_continuation_curve_save = false;
                preserve_partial_curves = true;
                check_duplicate_after_deflation = true;
                duplicate_after_deflation_retries = 2;
                duplicate_after_deflation_tolerance = T(1.0e-8);
                max_failed_continuations_per_knot = 3;
                failed_continuation_rejection_tolerance = T(1.0e-8);
                knot_relocation.set_default();
                seed_schedule.set_default();
                failed_candidate_registry.set_default();
                recovery_registry.set_default();
                topology_registry.set_default();
            }

            void plot_all()
            {
                std::cout << "||  |==allow_incomplete_restart_intersections: " << allow_incomplete_restart_intersections << std::endl;
                std::cout << "||  |==allow_knot_interpolation_failure: " << allow_knot_interpolation_failure << std::endl;
                std::cout << "||  |==allow_failed_continuation_curve_save: " << allow_failed_continuation_curve_save << std::endl;
                std::cout << "||  |==preserve_partial_curves: " << preserve_partial_curves << std::endl;
                std::cout << "||  |==check_duplicate_after_deflation: " << check_duplicate_after_deflation << std::endl;
                std::cout << "||  |==duplicate_after_deflation_retries: " << duplicate_after_deflation_retries << std::endl;
                std::cout << "||  |==duplicate_after_deflation_tolerance: " << duplicate_after_deflation_tolerance << std::endl;
                std::cout << "||  |==max_failed_continuations_per_knot: " << max_failed_continuations_per_knot << std::endl;
                std::cout << "||  |==failed_continuation_rejection_tolerance: " << failed_continuation_rejection_tolerance << std::endl;
                std::cout << "||  |==knot_relocation: " << std::endl;
                knot_relocation.plot_all();
                std::cout << "||  |==seed_schedule: " << std::endl;
                seed_schedule.plot_all();
                std::cout << "||  |==failed_candidate_registry: " << std::endl;
                failed_candidate_registry.plot_all();
                std::cout << "||  |==recovery_registry: " << std::endl;
                recovery_registry.plot_all();
                std::cout << "||  |==topology_registry: " << std::endl;
                topology_registry.plot_all();
            }
        };

        struct continuation_parameter_bounds_s
        {
            bool enabled;
            T minimum;
            T maximum;
            bool resolve_with_knot_registry;

            void set_default()
            {
                enabled = false;
                minimum = T(0);
                maximum = T(0);
                resolve_with_knot_registry = true;
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                if(enabled)
                {
                    std::cout << "||  |==minimum: " << minimum << std::endl;
                    std::cout << "||  |==maximum: " << maximum << std::endl;
                }
                std::cout << "||  |==resolve_with_knot_registry: "
                          << resolve_with_knot_registry << std::endl;
            }
        };

        struct boundary_refinement_policy_s
        {
            bool preserve_last_converged_point;

            void set_default()
            {
                preserve_last_converged_point = true;
            }

            void plot_all()
            {
                std::cout << "||  |==preserve_last_converged_point: "
                          << preserve_last_converged_point << std::endl;
            }
        };

        struct branch_intersection_policy_s
        {
            bool enabled;
            unsigned int signature_norm_index;
            T signature_tolerance;
            T state_tolerance;
            T minimum_step_fraction_from_start;
            bool detect_forward_approach;
            T maximum_forward_lookahead_steps;
            T maximum_forward_distance_step_ratio;
            T minimum_forward_distance_reduction_ratio;
            unsigned int forward_distance_extrapolation_power;
            T forward_refinement_step_factor;
            unsigned int maximum_forward_refinements;
            unsigned int minimum_forward_refinements_for_verification;
            T maximum_verified_forward_steps_ahead;
            T maximum_verified_forward_distance_step_ratio;
            bool localize_analytical_targets;
            T analytical_target_parameter_tolerance;
            bool verbose;

            void set_default()
            {
                enabled = false;
                signature_norm_index = 0;
                signature_tolerance = T(1.0e-6);
                state_tolerance = T(1.0e-8);
                minimum_step_fraction_from_start = T(1.0e-3);
                detect_forward_approach = false;
                maximum_forward_lookahead_steps = T(1);
                maximum_forward_distance_step_ratio = T(1);
                minimum_forward_distance_reduction_ratio = T(0.05);
                forward_distance_extrapolation_power = 1;
                forward_refinement_step_factor = T(0.25);
                maximum_forward_refinements = 8;
                minimum_forward_refinements_for_verification = 3;
                maximum_verified_forward_steps_ahead = T(0.1);
                maximum_verified_forward_distance_step_ratio = T(0.3);
                localize_analytical_targets = true;
                analytical_target_parameter_tolerance = T(1.0e-7);
                verbose = false;
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==signature_norm_index: " << signature_norm_index << std::endl;
                std::cout << "||  |==signature_tolerance: " << signature_tolerance << std::endl;
                std::cout << "||  |==state_tolerance: " << state_tolerance << std::endl;
                std::cout << "||  |==minimum_step_fraction_from_start: " << minimum_step_fraction_from_start << std::endl;
                std::cout << "||  |==detect_forward_approach: " << detect_forward_approach << std::endl;
                std::cout << "||  |==maximum_forward_lookahead_steps: " << maximum_forward_lookahead_steps << std::endl;
                std::cout << "||  |==maximum_forward_distance_step_ratio: " << maximum_forward_distance_step_ratio << std::endl;
                std::cout << "||  |==minimum_forward_distance_reduction_ratio: " << minimum_forward_distance_reduction_ratio << std::endl;
                std::cout << "||  |==forward_distance_extrapolation_power: " << forward_distance_extrapolation_power << std::endl;
                std::cout << "||  |==forward_refinement_step_factor: " << forward_refinement_step_factor << std::endl;
                std::cout << "||  |==maximum_forward_refinements: " << maximum_forward_refinements << std::endl;
                std::cout << "||  |==minimum_forward_refinements_for_verification: " << minimum_forward_refinements_for_verification << std::endl;
                std::cout << "||  |==maximum_verified_forward_steps_ahead: " << maximum_verified_forward_steps_ahead << std::endl;
                std::cout << "||  |==maximum_verified_forward_distance_step_ratio: " << maximum_verified_forward_distance_step_ratio << std::endl;
                std::cout << "||  |==localize_analytical_targets: " << localize_analytical_targets << std::endl;
                std::cout << "||  |==analytical_target_parameter_tolerance: " << analytical_target_parameter_tolerance << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };

        struct self_intersection_policy_s
        {
            bool enabled;
            unsigned int signature_norm_index;
            T signature_tolerance;
            T state_tolerance;
            T minimum_step_fraction_from_start;
            uint64_t minimum_index_gap;
            bool verbose;

            void set_default()
            {
                enabled = false;
                signature_norm_index = 0;
                signature_tolerance = T(1.0e-6);
                state_tolerance = T(1.0e-8);
                minimum_step_fraction_from_start = T(1.0e-3);
                minimum_index_gap = 50;
                verbose = false;
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==signature_norm_index: " << signature_norm_index << std::endl;
                std::cout << "||  |==signature_tolerance: " << signature_tolerance << std::endl;
                std::cout << "||  |==state_tolerance: " << state_tolerance << std::endl;
                std::cout << "||  |==minimum_step_fraction_from_start: " << minimum_step_fraction_from_start << std::endl;
                std::cout << "||  |==minimum_index_gap: " << minimum_index_gap << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };

        struct isotropy_transition_policy_s
        {
            bool enabled;
            T relative_mode_tolerance;
            std::size_t maximum_order;
            unsigned int maximum_refinements;
            T refinement_step_factor;
            T registry_lambda_tolerance;
            T registry_state_tolerance;
            bool verbose;

            void set_default()
            {
                enabled = false;
                relative_mode_tolerance = T(1.0e-10);
                maximum_order = 16;
                maximum_refinements = 6;
                refinement_step_factor = T(0.5);
                registry_lambda_tolerance = T(1.0e-7);
                registry_state_tolerance = T(1.0e-7);
                verbose = false;
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==relative_mode_tolerance: " << relative_mode_tolerance << std::endl;
                std::cout << "||  |==maximum_order: " << maximum_order << std::endl;
                std::cout << "||  |==maximum_refinements: " << maximum_refinements << std::endl;
                std::cout << "||  |==refinement_step_factor: " << refinement_step_factor << std::endl;
                std::cout << "||  |==registry_lambda_tolerance: " << registry_lambda_tolerance << std::endl;
                std::cout << "||  |==registry_state_tolerance: " << registry_state_tolerance << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };

        struct predictor_chart_policy_s
        {
            bool enabled;
            bool enforce;
            T weak_progress_warning_ratio;
            T minimum_progress_ratio;
            T progress_jump_warning_ratio;
            T maximum_progress_ratio;
            T displacement_warning_ratio;
            T maximum_displacement_ratio;
            unsigned int maximum_retries;
            T step_reduction_factor;

            void set_default()
            {
                enabled = true;
                enforce = true;
                weak_progress_warning_ratio = T(0.2);
                minimum_progress_ratio = T(0.02);
                progress_jump_warning_ratio = T(5);
                maximum_progress_ratio = T(50);
                displacement_warning_ratio = T(20);
                maximum_displacement_ratio = T(100);
                maximum_retries = 4;
                step_reduction_factor = T(0.2);
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==enforce: " << enforce << std::endl;
                std::cout << "||  |==weak_progress_warning_ratio: " << weak_progress_warning_ratio << std::endl;
                std::cout << "||  |==minimum_progress_ratio: " << minimum_progress_ratio << std::endl;
                std::cout << "||  |==progress_jump_warning_ratio: " << progress_jump_warning_ratio << std::endl;
                std::cout << "||  |==maximum_progress_ratio: " << maximum_progress_ratio << std::endl;
                std::cout << "||  |==displacement_warning_ratio: " << displacement_warning_ratio << std::endl;
                std::cout << "||  |==maximum_displacement_ratio: " << maximum_displacement_ratio << std::endl;
                std::cout << "||  |==maximum_retries: " << maximum_retries << std::endl;
                std::cout << "||  |==step_reduction_factor: " << step_reduction_factor << std::endl;
            }
        };

        struct corrector_retry_policy_s
        {
            unsigned int maximum_retries;
            T failure_reduction_factor;
            T minimum_step_size;
            T minimum_step_ratio;
            unsigned int successes_before_growth;
            T success_growth_factor;

            void set_default()
            {
                maximum_retries = 8;
                failure_reduction_factor = T(0.5);
                minimum_step_size = T(0);
                minimum_step_ratio = T(1.0e-6);
                successes_before_growth = 5;
                success_growth_factor = T(1.25);
            }

            void plot_all()
            {
                std::cout << "||  |==maximum_retries: " << maximum_retries << std::endl;
                std::cout << "||  |==failure_reduction_factor: " << failure_reduction_factor << std::endl;
                std::cout << "||  |==minimum_step_size: " << minimum_step_size << std::endl;
                std::cout << "||  |==minimum_step_ratio: " << minimum_step_ratio << std::endl;
                std::cout << "||  |==successes_before_growth: " << successes_before_growth << std::endl;
                std::cout << "||  |==success_growth_factor: " << success_growth_factor << std::endl;
            }
        };

        struct progress_monitor_policy_s
        {
            bool enabled;
            unsigned int window_size;
            T minimum_window_progress_ratio;

            void set_default()
            {
                enabled = false;
                window_size = 25;
                minimum_window_progress_ratio = T(1.0e-3);
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==window_size: " << window_size << std::endl;
                std::cout << "||  |==minimum_window_progress_ratio: "
                          << minimum_window_progress_ratio << std::endl;
            }
        };

        unsigned int   continuation_steps;
        T              step_size;
        T              max_step_size;
        unsigned int   deflation_attempts;
        unsigned int   continuation_fail_attempts;
        int            initial_direciton;
        T              step_ds_m;
        T              step_ds_p;
        unsigned int   skip_files;
        std::vector<T> deflation_knots;
        bool           add_analytical_solution_to_diagram;
        std::vector<unsigned int> analytical_solution_branches;

        continuation_parameter_bounds_s continuation_parameter_bounds;
        boundary_refinement_policy_s    boundary_refinement_policy;
        restart_policy_s                 restart_policy;
        branch_intersection_policy_s     branch_intersection_policy;
        self_intersection_policy_s       self_intersection_policy;
        isotropy_transition_policy_s     isotropy_transition_policy;
        predictor_chart_policy_s         predictor_chart_policy;
        corrector_retry_policy_s         corrector_retry_policy;
        progress_monitor_policy_s        progress_monitor_policy;
        linear_solver_extended_s       linear_solver_extended;
        newton_extended_continuation_s newton_extended_continuation;
        newton_extended_deflation_s    newton_extended_deflation;

        void set_default()
        {
            continuation_steps         = 5000;
            step_size                  = 0.5;
            max_step_size              = 5.0;
            deflation_attempts         = 5;
            continuation_fail_attempts = 4;
            initial_direciton          = -1;
            step_ds_m                  = 0.2;
            step_ds_p                  = 0.01;
            skip_files                 = 100;
            deflation_knots            = { 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
            add_analytical_solution_to_diagram = false;
            analytical_solution_branches = {};
            continuation_parameter_bounds.set_default();
            boundary_refinement_policy.set_default();
            restart_policy.set_default();
            branch_intersection_policy.set_default();
            self_intersection_policy.set_default();
            isotropy_transition_policy.set_default();
            predictor_chart_policy.set_default();
            corrector_retry_policy.set_default();
            progress_monitor_policy.set_default();
            linear_solver_extended.set_default();
            newton_extended_continuation.set_default();
            newton_extended_deflation.set_default();
        }
        void plot_all()
        {
            std::cout << "||==continuation_steps: " << continuation_steps << std::endl;
            std::cout << "||==step_size: " << step_size << std::endl;
            std::cout << "||==max_step_size: " << max_step_size << std::endl;
            std::cout << "||==deflation_attempts: " << deflation_attempts << std::endl;
            std::cout << "||==continuation_fail_attempts: " << continuation_fail_attempts << std::endl;
            std::cout << "||==initial_direciton: " << initial_direciton << std::endl;
            std::cout << "||==step_ds_m: " << step_ds_m << std::endl;
            std::cout << "||==step_ds_p: " << step_ds_p << std::endl;
            std::cout << "||==skip_files: " << skip_files << std::endl;
            std::cout << "||==deflation_knots: ";
            for ( auto &x : deflation_knots )
                std::cout << x << " ";
            std::cout << std::endl;
            std::cout << "||==add_analytical_solution_to_diagram: " << add_analytical_solution_to_diagram << std::endl;
            std::cout << "||==analytical_solution_branches: ";
            if(analytical_solution_branches.empty())
            {
                std::cout << "all";
            }
            else
            {
                for(auto &x: analytical_solution_branches)
                    std::cout << x << " ";
            }
            std::cout << std::endl;
            std::cout << "||==continuation_parameter_bounds: " << std::endl;
            continuation_parameter_bounds.plot_all();
            std::cout << "||==boundary_refinement_policy: " << std::endl;
            boundary_refinement_policy.plot_all();
            std::cout << "||==restart_policy: " << std::endl;
            restart_policy.plot_all();
            std::cout << "||==branch_intersection_policy: " << std::endl;
            branch_intersection_policy.plot_all();
            std::cout << "||==self_intersection_policy: " << std::endl;
            self_intersection_policy.plot_all();
            std::cout << "||==isotropy_transition_policy: " << std::endl;
            isotropy_transition_policy.plot_all();
            std::cout << "||==predictor_chart_policy: " << std::endl;
            predictor_chart_policy.plot_all();
            std::cout << "||==corrector_retry_policy: " << std::endl;
            corrector_retry_policy.plot_all();
            std::cout << "||==progress_monitor_policy: " << std::endl;
            progress_monitor_policy.plot_all();
            std::cout << "||==linear_solver_extended: " << std::endl;
            linear_solver_extended.plot_all();
            std::cout << "||==newton_extended_continuation: " << std::endl;
            newton_extended_continuation.plot_all();
            std::cout << "||==newton_extended_deflation: " << std::endl;
            newton_extended_deflation.plot_all();
        }
    };

    struct stability_continuation_s
    {

        struct linear_solver_s
        {
            unsigned int lin_solver_max_it;
            unsigned int use_precond_resid;
            unsigned int resid_recalc_freq;
            unsigned int basis_size;
            T lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
                              //those are custom parameters to be set only to high dim Krylov methods
            bool save_convergence_history;
            bool divide_out_norms_by_rel_base;
            bool verbose;

            void set_default()
            {
                lin_solver_max_it = 1500;
                use_precond_resid = 1;
                resid_recalc_freq = 1;
                basis_size        = 4;
                lin_solver_tol =
                    5.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
                            //those are custom parameters to be set only to high dim Krylov methods
                save_convergence_history     = true;
                divide_out_norms_by_rel_base = true;
                verbose                      = true;
            }

            void plot_all()
            {
                std::cout << "||  |==lin_solver_max_it: " << lin_solver_max_it << std::endl;
                std::cout << "||  |==use_precond_resid: " << use_precond_resid << std::endl;
                std::cout << "||  |==resid_recalc_freq: " << resid_recalc_freq << std::endl;
                std::cout << "||  |==basis_size: " << basis_size << std::endl;
                std::cout << "||  |==lin_solver_tol: " << lin_solver_tol << std::endl;
                std::cout << "||  |==save_convergence_history: " << save_convergence_history << std::endl;
                std::cout << "||  |==divide_out_norms_by_rel_base: " << divide_out_norms_by_rel_base << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };

        struct newton_s
        {
            unsigned int newton_max_it;
            T            newton_wight;
            bool         store_norms_history;
            bool         verbose;
            T            tolerance;

            void set_default()
            {
                newton_max_it       = 300;
                newton_wight        = T( 1.0 );
                store_norms_history = true;
                verbose             = true;
                tolerance           = 1.0e-9;
            }
            void plot_all()
            {
                std::cout << "||  |==newton_max_it: " << newton_max_it << std::endl;
                std::cout << "||  |==newton_wight: " << newton_wight << std::endl;
                std::cout << "||  |==store_norms_history: " << store_norms_history << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
                std::cout << "||  |==tolerance: " << tolerance << std::endl;
            }
        };

        bool            linear_operator_stable_eigenvalues_left_halfplane;
        unsigned int    Krylov_subspace;
        unsigned int    desired_spectrum;
        std::vector<T>  Cayley_transform_sigma_mu;
        T               stability_boundary_tolerance;
        T               real_eigenvalue_tolerance;
        T               conjugate_pair_tolerance;
        bool            require_converged_eigenpairs;
        bool            require_nonempty_spectrum;
        bool            require_complete_scan_coverage;
        unsigned int    spectrum_classification_retries;
        unsigned int    transition_classification_confirmations;
        bool            correct_stability_transitions_with_newton;
        unsigned int    transition_refinement_maximum_iterations;
        unsigned int    transition_refinement_maximum_subdivisions;
        T               transition_refinement_parameter_tolerance;
        unsigned int    symmetry_endpoint_guard_source_points;
        stability::analysis::matrix_free_stability_config<T>
                        matrix_free_eigensolver;
        linear_solver_s linear_solver;
        newton_s        newton;

        void set_default()
        {
            linear_operator_stable_eigenvalues_left_halfplane = true;
            Krylov_subspace                                   = 25;
            desired_spectrum                                  = 5;
            stability_boundary_tolerance                       = T(1.0e-7);
            real_eigenvalue_tolerance                          = T(1.0e-7);
            conjugate_pair_tolerance                           = T(1.0e-6);
            require_converged_eigenpairs                       = true;
            require_nonempty_spectrum                          = true;
            require_complete_scan_coverage                     = true;
            spectrum_classification_retries                    = 2;
            transition_classification_confirmations            = 2;
            correct_stability_transitions_with_newton          = false;
            transition_refinement_maximum_iterations           = 20;
            transition_refinement_maximum_subdivisions          = 8;
            transition_refinement_parameter_tolerance          = T(0);
            symmetry_endpoint_guard_source_points               = 0;
            matrix_free_eigensolver                            = {};
            linear_solver.set_default();
            newton.set_default();
        }
        void plot_all()
        {
            std::cout << "||==linear_operator_stable_eigenvalues_left_halfplane: "
                      << linear_operator_stable_eigenvalues_left_halfplane << std::endl;
            std::cout << "||==Krylov_subspace: " << Krylov_subspace << std::endl;
            std::cout << "||==desired_spectrum: " << desired_spectrum << std::endl;
            std::cout << "||==Cayley_transform_sigma_mu: ";
            for ( auto &x_ : Cayley_transform_sigma_mu )
            {
                std::cout << x_ << " ";
            }
            std::cout << std::endl;
            std::cout << "||==stability_boundary_tolerance: "
                      << stability_boundary_tolerance << std::endl;
            std::cout << "||==real_eigenvalue_tolerance: "
                      << real_eigenvalue_tolerance << std::endl;
            std::cout << "||==conjugate_pair_tolerance: "
                      << conjugate_pair_tolerance << std::endl;
            std::cout << "||==require_converged_eigenpairs: "
                      << require_converged_eigenpairs << std::endl;
            std::cout << "||==require_nonempty_spectrum: "
                      << require_nonempty_spectrum << std::endl;
            std::cout << "||==require_complete_scan_coverage: "
                      << require_complete_scan_coverage << std::endl;
            std::cout << "||==spectrum_classification_retries: "
                      << spectrum_classification_retries << std::endl;
            std::cout << "||==transition_classification_confirmations: "
                      << transition_classification_confirmations
                      << std::endl;
            std::cout << "||==correct_stability_transitions_with_newton: "
                      << correct_stability_transitions_with_newton
                      << std::endl;
            std::cout << "||==transition_refinement_maximum_iterations: "
                      << transition_refinement_maximum_iterations
                      << std::endl;
            std::cout << "||==transition_refinement_maximum_subdivisions: "
                      << transition_refinement_maximum_subdivisions
                      << std::endl;
            std::cout << "||==transition_refinement_parameter_tolerance: "
                      << transition_refinement_parameter_tolerance
                      << std::endl;
            std::cout << "||==symmetry_endpoint_guard_source_points: "
                      << symmetry_endpoint_guard_source_points
                      << std::endl;
            std::cout << "||==matrix_free_eigensolver: "
                      << (matrix_free_eigensolver.enabled
                              ? "enabled"
                              : "disabled")
                      << std::endl;
            if(matrix_free_eigensolver.enabled)
            {
                const auto& config = matrix_free_eigensolver;
                std::cout
                    << "||  |==linearization_scale: "
                    << config.linearization_scale
                    << std::endl;
                std::cout
                    << "||  |==transformation: "
                    << stability::analysis::
                           matrix_free_spectral_transformation_name(
                               config.transformation.type)
                    << std::endl;
                std::cout
                    << "||  |==shifts:";
                for(const auto& shift :
                    config.transformation.shifts)
                {
                    std::cout
                        << " (" << shift.real()
                        << "," << shift.imag() << ")";
                }
                std::cout << std::endl;
                std::cout
                    << "||  |==desired_eigenvalues: "
                    << config.outer.desired_eigenvalues
                    << std::endl;
                std::cout
                    << "||  |==krylov_dimension: "
                    << config.outer.krylov_dimension
                    << std::endl;
                std::cout
                    << "||  |==inner_basis_size: "
                    << config.inner_solver.basis_size
                    << std::endl;
                std::cout
                    << "||  |==inner_preconditioner_side: "
                    << config.inner_solver.preconditioner_side
                    << std::endl;
                std::cout
                    << "||  |==inner_basis_retry_sizes:";
                for(const auto basis_size :
                    config.inner_solver.basis_retry_sizes)
                {
                    std::cout << " " << basis_size;
                }
                std::cout << std::endl;
                std::cout
                    << "||  |==scan_retry: "
                    << (config.retry.enabled
                            ? "enabled"
                            : "disabled")
                    << ", maximum_shift_retries = "
                    << config.retry.maximum_shift_retries
                    << ", initial_shift_perturbation = "
                    << config.retry.initial_shift_perturbation
                    << std::endl;
                std::cout
                    << "||  |==preconditioner_pole_tolerances: "
                    << config.retry.
                           preconditioner_pole_absolute_tolerance
                    << " "
                    << config.retry.
                           preconditioner_pole_relative_tolerance
                    << std::endl;
                std::cout
                    << "||  |==multiplicity_probe_count: "
                    << config.aggregation.probe_count
                    << std::endl;
                std::cout
                    << "||  |==small_system: "
                    << (config.small_system.enabled
                            ? "enabled"
                            : "disabled")
                    << ", maximum_dimension = "
                    << config.small_system.maximum_dimension
                    << ", prefer = "
                    << config.small_system.prefer
                    << std::endl;
                std::cout
                    << "||  |==eigenvector_independence_tolerance: "
                    << config.aggregation.
                           eigenvector_independence_tolerance
                    << std::endl;
            }
            std::cout << "||==linear_solver: " << std::endl;
            linear_solver.plot_all();
            std::cout << "||==newton: " << std::endl;
            newton.plot_all();
        }
    };

    struct nonlinear_operator_s
    {

        struct linear_solver_s
        {
            unsigned int lin_solver_max_it;
            unsigned int use_precond_resid;
            unsigned int resid_recalc_freq;
            unsigned int basis_size;
            T    lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
            bool save_convergence_history;
            bool divide_out_norms_by_rel_base;
            bool verbose;

            void set_default()
            {
                lin_solver_max_it = 1500;
                use_precond_resid = 1;
                resid_recalc_freq = 1;
                basis_size        = 4;
                lin_solver_tol =
                    5.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
                save_convergence_history     = true;
                divide_out_norms_by_rel_base = true;
                verbose                      = true;
            }

            void plot_all()
            {
                std::cout << "||  |==lin_solver_max_it: " << lin_solver_max_it << std::endl;
                std::cout << "||  |==use_precond_resid: " << use_precond_resid << std::endl;
                std::cout << "||  |==resid_recalc_freq: " << resid_recalc_freq << std::endl;
                std::cout << "||  |==basis_size: " << basis_size << std::endl;
                std::cout << "||  |==lin_solver_tol: " << lin_solver_tol << std::endl;
                std::cout << "||  |==save_convergence_history: " << save_convergence_history << std::endl;
                std::cout << "||  |==divide_out_norms_by_rel_base: " << divide_out_norms_by_rel_base << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
            }
        };

        struct newton_s
        {
            unsigned int newton_max_it;
            T            newton_wight;
            bool         store_norms_history;
            bool         verbose;
            T            tolerance;

            void set_default()
            {
                newton_max_it       = 300;
                newton_wight        = T( 1.0 );
                store_norms_history = true;
                verbose             = true;
                tolerance           = 1.0e-9;
            }

            void plot_all()
            {
                std::cout << "||  |==newton_max_it: " << newton_max_it << std::endl;
                std::cout << "||  |==newton_wight: " << newton_wight << std::endl;
                std::cout << "||  |==store_norms_history: " << store_norms_history << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
                std::cout << "||  |==tolerance: " << tolerance << std::endl;
            }
        };

        std::vector<size_t> N_size;
        // problem dependent:
        std::vector<T>   problem_real_parameters_vector;
        std::vector<int> problem_int_parameters_vector;

        linear_solver_s linear_solver;
        newton_s        newton;

        void set_default()
        {
            N_size = { 256, 256 };

            problem_real_parameters_vector = { 0.5, 2.0, 4.0 };
            problem_int_parameters_vector  = { 2 };

            linear_solver.set_default();
            newton.set_default();
        }
        void plot_all()
        {
            std::cout << "||==N_size: ";
            for ( auto &x : N_size )
                std::cout << x << " ";
            std::cout << std::endl;
            std::cout << "||==problem_real_parameters_vector: ";
            for ( auto &x : problem_real_parameters_vector )
                std::cout << x << " ";
            std::cout << std::endl;
            std::cout << "||==problem_int_parameters_vector: ";
            for ( auto &x : problem_int_parameters_vector )
                std::cout << x << " ";
            std::cout << std::endl;
            std::cout << "||==linear_solver: " << std::endl;
            linear_solver.plot_all();
            std::cout << "||==newton: " << std::endl;
            newton.plot_all();
        }
    };

    struct plot_solutions_s
    {
        int plot_solution_frequency;

        void set_default()
        {
            plot_solution_frequency = 3;
        }
        void plot_all()
        {
            std::cout << "||==plot_solution_frequency: " << plot_solution_frequency << std::endl;
        }
    };

    int         nvidia_pci_id;
    bool        use_high_precision_reduction;
    std::string path_to_project; //relative to the execution root directory
    //for serialization, just the filenames
    std::string bifurcaiton_diagram_file_name;
    std::string stability_diagram_file_name;


    deflation_continuation_s deflation_continuation;
    stability_continuation_s stability_continuation;
    nonlinear_operator_s     nonlinear_operator;
    plot_solutions_s         plot_solutions;

    void set_default()
    {
        nvidia_pci_id                 = -1;
        use_high_precision_reduction  = false;
        path_to_project                = "./";
        bifurcaiton_diagram_file_name = "bifurcation_diagram.dat";
        stability_diagram_file_name   = "stability_diagram.dat";

        deflation_continuation.set_default();
        stability_continuation.set_default();
        nonlinear_operator.set_default();
        plot_solutions.set_default();
    }

    void plot_all()
    {
        std::cout << std::endl;
        std::cout << "nvidia_pci_id: " << nvidia_pci_id << std::endl;
        std::cout << "use_high_precision_reduction: " << use_high_precision_reduction << std::endl;
        std::cout << "path_to_project: " << path_to_project << std::endl;
        std::cout << "bifurcaiton_diagram_file_name: " << bifurcaiton_diagram_file_name << std::endl;
        std::cout << "stability_diagram_file_name: " << stability_diagram_file_name << std::endl;
        std::cout << "deflation_continuation: " << std::endl;
        deflation_continuation.plot_all();
        std::cout << "stability_continuation: " << std::endl;
        stability_continuation.plot_all();
        std::cout << "nonlinear_operator: " << std::endl;
        nonlinear_operator.plot_all();
        std::cout << "plot_solutions: " << std::endl;
        plot_solutions.plot_all();
    }
};


typedef parameters<double> parameters_d;
typedef parameters<float>  parameters_f;

}
#endif // __MAIN_PARAMETER_TYPES_H__
