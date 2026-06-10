#ifndef __PARAMETERS_HPP__
#define __PARAMETERS_HPP__

/**
*
* Main structure to hold all parameters of execution.
* It is to be filled by the json config file from each project.
*
*/

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

//includes json library by nlohmann
#include <contrib/json/nlohmann/json.hpp>

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
                bool enabled;
                std::string registry_file;
                T min_shift_abs;
                T max_shift_abs;
                unsigned int candidate_count;
                bool prefer_positive_shift;
                bool require_all_intersections;
                bool save_registry;

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
                }
            };

            bool allow_incomplete_restart_intersections;
            bool allow_knot_interpolation_failure;
            bool allow_failed_continuation_curve_save;
            bool check_duplicate_after_deflation;
            unsigned int duplicate_after_deflation_retries;
            T duplicate_after_deflation_tolerance;
            unsigned int max_failed_continuations_per_knot;
            T failed_continuation_rejection_tolerance;
            knot_relocation_s knot_relocation;

            void set_default()
            {
                allow_incomplete_restart_intersections = false;
                allow_knot_interpolation_failure = false;
                allow_failed_continuation_curve_save = false;
                check_duplicate_after_deflation = true;
                duplicate_after_deflation_retries = 2;
                duplicate_after_deflation_tolerance = T(1.0e-8);
                max_failed_continuations_per_knot = 3;
                failed_continuation_rejection_tolerance = T(1.0e-8);
                knot_relocation.set_default();
            }

            void plot_all()
            {
                std::cout << "||  |==allow_incomplete_restart_intersections: " << allow_incomplete_restart_intersections << std::endl;
                std::cout << "||  |==allow_knot_interpolation_failure: " << allow_knot_interpolation_failure << std::endl;
                std::cout << "||  |==allow_failed_continuation_curve_save: " << allow_failed_continuation_curve_save << std::endl;
                std::cout << "||  |==check_duplicate_after_deflation: " << check_duplicate_after_deflation << std::endl;
                std::cout << "||  |==duplicate_after_deflation_retries: " << duplicate_after_deflation_retries << std::endl;
                std::cout << "||  |==duplicate_after_deflation_tolerance: " << duplicate_after_deflation_tolerance << std::endl;
                std::cout << "||  |==max_failed_continuations_per_knot: " << max_failed_continuations_per_knot << std::endl;
                std::cout << "||  |==failed_continuation_rejection_tolerance: " << failed_continuation_rejection_tolerance << std::endl;
                std::cout << "||  |==knot_relocation: " << std::endl;
                knot_relocation.plot_all();
            }
        };

        struct branch_intersection_policy_s
        {
            bool enabled;
            unsigned int signature_norm_index;
            T signature_tolerance;
            T state_tolerance;
            T minimum_step_fraction_from_start;
            bool verbose;

            void set_default()
            {
                enabled = false;
                signature_norm_index = 0;
                signature_tolerance = T(1.0e-6);
                state_tolerance = T(1.0e-8);
                minimum_step_fraction_from_start = T(1.0e-3);
                verbose = false;
            }

            void plot_all()
            {
                std::cout << "||  |==enabled: " << enabled << std::endl;
                std::cout << "||  |==signature_norm_index: " << signature_norm_index << std::endl;
                std::cout << "||  |==signature_tolerance: " << signature_tolerance << std::endl;
                std::cout << "||  |==state_tolerance: " << state_tolerance << std::endl;
                std::cout << "||  |==minimum_step_fraction_from_start: " << minimum_step_fraction_from_start << std::endl;
                std::cout << "||  |==verbose: " << verbose << std::endl;
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

        restart_policy_s                 restart_policy;
        branch_intersection_policy_s     branch_intersection_policy;
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
            restart_policy.set_default();
            branch_intersection_policy.set_default();
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
            std::cout << "||==restart_policy: " << std::endl;
            restart_policy.plot_all();
            std::cout << "||==branch_intersection_policy: " << std::endl;
            branch_intersection_policy.plot_all();
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
        linear_solver_s linear_solver;
        newton_s        newton;

        void set_default()
        {
            linear_operator_stable_eigenvalues_left_halfplane = true;
            Krylov_subspace                                   = 25;
            desired_spectrum                                  = 5;
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


void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::linear_solver_extended_s &params_dc_lse_
)
{
    params_dc_lse_ = parameters_d::deflation_continuation_s::linear_solver_extended_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<double>(),
        j.at( "use_small_alpha_approximation" ).get<bool>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}
void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::linear_solver_extended_s &params_dc_lse_
)
{
    params_dc_lse_ = parameters_f::deflation_continuation_s::linear_solver_extended_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<float>(),
        j.at( "use_small_alpha_approximation" ).get<bool>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}


void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::newton_extended_continuation_s &params_dc_nec_
)
{
    params_dc_nec_ = parameters_d::deflation_continuation_s::newton_extended_continuation_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "update_wight_maximum" ).get<double>(),
        j.at( "save_norms_history" ).get<bool>(),
        j.at( "verbose" ).get<bool>(),
        j.at( "tolerance" ).get<double>(),
        j.at( "relax_tolerance_factor" ).get<double>(),
        j.at( "relax_tolerance_steps" ).get<int>(),
        j.at( "stagnation_max" ).get<unsigned int>(),
        j.at( "maximum_norm_increase" ).get<double>(),
        j.at( "newton_wight_threshold" ).get<double>()

    };
}
void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::newton_extended_continuation_s &params_dc_nec_
)
{
    params_dc_nec_ = parameters_f::deflation_continuation_s::newton_extended_continuation_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "update_wight_maximum" ).get<float>(),
        j.at( "save_norms_history" ).get<bool>(),
        j.at( "verbose" ).get<bool>(),
        j.at( "tolerance" ).get<float>(),
        j.at( "relax_tolerance_factor" ).get<float>(),
        j.at( "relax_tolerance_steps" ).get<int>(),
        j.at( "stagnation_max" ).get<unsigned int>(),
        j.at( "maximum_norm_increase" ).get<float>(),
        j.at( "newton_wight_threshold" ).get<float>()

    };
}

void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::newton_extended_deflation_s &params_dc_ned_
)
{
    params_dc_ned_ = parameters_d::deflation_continuation_s::newton_extended_deflation_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<double>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<double>()

    };
}
void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::newton_extended_deflation_s &params_dc_ned_
)
{
    params_dc_ned_ = parameters_f::deflation_continuation_s::newton_extended_deflation_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<float>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<float>()

    };
}

void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::restart_policy_s::knot_relocation_s &params_dc_kr_
)
{
    params_dc_kr_.set_default();
    params_dc_kr_.enabled = j.value( "enabled", params_dc_kr_.enabled );
    params_dc_kr_.registry_file = j.value( "registry_file", params_dc_kr_.registry_file );
    params_dc_kr_.min_shift_abs = j.value( "min_shift_abs", params_dc_kr_.min_shift_abs );
    params_dc_kr_.max_shift_abs = j.value( "max_shift_abs", params_dc_kr_.max_shift_abs );
    params_dc_kr_.candidate_count = j.value( "candidate_count", params_dc_kr_.candidate_count );
    params_dc_kr_.prefer_positive_shift = j.value( "prefer_positive_shift", params_dc_kr_.prefer_positive_shift );
    params_dc_kr_.require_all_intersections = j.value( "require_all_intersections", params_dc_kr_.require_all_intersections );
    params_dc_kr_.save_registry = j.value( "save_registry", params_dc_kr_.save_registry );
}

void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::restart_policy_s::knot_relocation_s &params_dc_kr_
)
{
    params_dc_kr_.set_default();
    params_dc_kr_.enabled = j.value( "enabled", params_dc_kr_.enabled );
    params_dc_kr_.registry_file = j.value( "registry_file", params_dc_kr_.registry_file );
    params_dc_kr_.min_shift_abs = j.value( "min_shift_abs", params_dc_kr_.min_shift_abs );
    params_dc_kr_.max_shift_abs = j.value( "max_shift_abs", params_dc_kr_.max_shift_abs );
    params_dc_kr_.candidate_count = j.value( "candidate_count", params_dc_kr_.candidate_count );
    params_dc_kr_.prefer_positive_shift = j.value( "prefer_positive_shift", params_dc_kr_.prefer_positive_shift );
    params_dc_kr_.require_all_intersections = j.value( "require_all_intersections", params_dc_kr_.require_all_intersections );
    params_dc_kr_.save_registry = j.value( "save_registry", params_dc_kr_.save_registry );
}

void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::restart_policy_s &params_dc_rp_
)
{
    params_dc_rp_.set_default();
    params_dc_rp_.allow_incomplete_restart_intersections =
        j.value( "allow_incomplete_restart_intersections", params_dc_rp_.allow_incomplete_restart_intersections );
    params_dc_rp_.allow_knot_interpolation_failure =
        j.value( "allow_knot_interpolation_failure", params_dc_rp_.allow_knot_interpolation_failure );
    params_dc_rp_.allow_failed_continuation_curve_save =
        j.value( "allow_failed_continuation_curve_save", params_dc_rp_.allow_failed_continuation_curve_save );
    params_dc_rp_.check_duplicate_after_deflation =
        j.value( "check_duplicate_after_deflation", params_dc_rp_.check_duplicate_after_deflation );
    params_dc_rp_.duplicate_after_deflation_retries =
        j.value( "duplicate_after_deflation_retries", params_dc_rp_.duplicate_after_deflation_retries );
    params_dc_rp_.duplicate_after_deflation_tolerance =
        j.value( "duplicate_after_deflation_tolerance", params_dc_rp_.duplicate_after_deflation_tolerance );
    params_dc_rp_.max_failed_continuations_per_knot =
        j.value( "max_failed_continuations_per_knot", params_dc_rp_.max_failed_continuations_per_knot );
    params_dc_rp_.failed_continuation_rejection_tolerance =
        j.value( "failed_continuation_rejection_tolerance", params_dc_rp_.failed_continuation_rejection_tolerance );
    params_dc_rp_.knot_relocation =
        j.value( "knot_relocation", nlohmann::json::object() ).get<parameters_d::deflation_continuation_s::restart_policy_s::knot_relocation_s>();
}

void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::restart_policy_s &params_dc_rp_
)
{
    params_dc_rp_.set_default();
    params_dc_rp_.allow_incomplete_restart_intersections =
        j.value( "allow_incomplete_restart_intersections", params_dc_rp_.allow_incomplete_restart_intersections );
    params_dc_rp_.allow_knot_interpolation_failure =
        j.value( "allow_knot_interpolation_failure", params_dc_rp_.allow_knot_interpolation_failure );
    params_dc_rp_.allow_failed_continuation_curve_save =
        j.value( "allow_failed_continuation_curve_save", params_dc_rp_.allow_failed_continuation_curve_save );
    params_dc_rp_.check_duplicate_after_deflation =
        j.value( "check_duplicate_after_deflation", params_dc_rp_.check_duplicate_after_deflation );
    params_dc_rp_.duplicate_after_deflation_retries =
        j.value( "duplicate_after_deflation_retries", params_dc_rp_.duplicate_after_deflation_retries );
    params_dc_rp_.duplicate_after_deflation_tolerance =
        j.value( "duplicate_after_deflation_tolerance", params_dc_rp_.duplicate_after_deflation_tolerance );
    params_dc_rp_.max_failed_continuations_per_knot =
        j.value( "max_failed_continuations_per_knot", params_dc_rp_.max_failed_continuations_per_knot );
    params_dc_rp_.failed_continuation_rejection_tolerance =
        j.value( "failed_continuation_rejection_tolerance", params_dc_rp_.failed_continuation_rejection_tolerance );
    params_dc_rp_.knot_relocation =
        j.value( "knot_relocation", nlohmann::json::object() ).get<parameters_f::deflation_continuation_s::restart_policy_s::knot_relocation_s>();
}

void from_json(
    const nlohmann::json &j, parameters_d::deflation_continuation_s::branch_intersection_policy_s &params_dc_bip_
)
{
    params_dc_bip_.set_default();
    params_dc_bip_.enabled = j.value( "enabled", params_dc_bip_.enabled );
    params_dc_bip_.signature_norm_index = j.value( "signature_norm_index", params_dc_bip_.signature_norm_index );
    params_dc_bip_.signature_tolerance = j.value( "signature_tolerance", params_dc_bip_.signature_tolerance );
    params_dc_bip_.state_tolerance = j.value( "state_tolerance", params_dc_bip_.state_tolerance );
    params_dc_bip_.minimum_step_fraction_from_start =
        j.value( "minimum_step_fraction_from_start", params_dc_bip_.minimum_step_fraction_from_start );
    params_dc_bip_.verbose = j.value( "verbose", params_dc_bip_.verbose );
}

void from_json(
    const nlohmann::json &j, parameters_f::deflation_continuation_s::branch_intersection_policy_s &params_dc_bip_
)
{
    params_dc_bip_.set_default();
    params_dc_bip_.enabled = j.value( "enabled", params_dc_bip_.enabled );
    params_dc_bip_.signature_norm_index = j.value( "signature_norm_index", params_dc_bip_.signature_norm_index );
    params_dc_bip_.signature_tolerance = j.value( "signature_tolerance", params_dc_bip_.signature_tolerance );
    params_dc_bip_.state_tolerance = j.value( "state_tolerance", params_dc_bip_.state_tolerance );
    params_dc_bip_.minimum_step_fraction_from_start =
        j.value( "minimum_step_fraction_from_start", params_dc_bip_.minimum_step_fraction_from_start );
    params_dc_bip_.verbose = j.value( "verbose", params_dc_bip_.verbose );
}


void from_json( const nlohmann::json &j, parameters_d::deflation_continuation_s &params_dc_ )
{
    // unsigned int continuation_steps;
    // T step_size;
    // unsigned int deflation_attempts;
    // unsigned int continuation_fail_attempts;
    // int initial_direciton;
    // T step_ds_m;
    // T step_ds_p;
    // unsigned int skip_files;
    // std::vector<T> deflation_knots;
    params_dc_ = parameters_d::deflation_continuation_s{
        j.at( "maximum_continuation_steps" ).get<unsigned int>(),
        j.at( "step_size" ).get<double>(),
        j.at( "max_step_size" ).get<double>(),
        j.at( "deflation_attempts" ).get<unsigned int>(),
        j.at( "continuation_fail_attempts" ).get<unsigned int>(),
        j.at( "initial_direciton" ).get<int>(),
        j.at( "minimum_step_multiplier" ).get<double>(),
        j.at( "maximum_step_multiplier" ).get<double>(),
        j.at( "skip_file_output" ).get<unsigned int>(),
        j.at( "deflation_knots" ).get<std::vector<double>>(),
        j.value( "restart_policy", nlohmann::json::object() ).get<parameters_d::deflation_continuation_s::restart_policy_s>(),
        j.value( "branch_intersection_policy", nlohmann::json::object() ).get<parameters_d::deflation_continuation_s::branch_intersection_policy_s>(),

        j.at( "linear_solver_extended" ).get<parameters_d::deflation_continuation_s::linear_solver_extended_s>(),
        j.at( "newton_continuation" ).get<parameters_d::deflation_continuation_s::newton_extended_continuation_s>(),
        j.at( "newton_deflation" ).get<parameters_d::deflation_continuation_s::newton_extended_deflation_s>()
    };
}
void from_json( const nlohmann::json &j, parameters_f::deflation_continuation_s &params_dc_ )
{
    params_dc_ = parameters_f::deflation_continuation_s{
        j.at( "maximum_continuation_steps" ).get<unsigned int>(),
        j.at( "step_size" ).get<float>(),
        j.at( "max_step_size" ).get<float>(),
        j.at( "deflation_attempts" ).get<unsigned int>(),
        j.at( "continuation_fail_attempts" ).get<unsigned int>(),
        j.at( "initial_direciton" ).get<int>(),
        j.at( "minimum_step_multiplier" ).get<float>(),
        j.at( "maximum_step_multiplier" ).get<float>(),
        j.at( "skip_file_output" ).get<unsigned int>(),
        j.at( "deflation_knots" ).get<std::vector<float>>(),
        j.value( "restart_policy", nlohmann::json::object() ).get<parameters_f::deflation_continuation_s::restart_policy_s>(),
        j.value( "branch_intersection_policy", nlohmann::json::object() ).get<parameters_f::deflation_continuation_s::branch_intersection_policy_s>(),

        j.at( "linear_solver_extended" ).get<parameters_f::deflation_continuation_s::linear_solver_extended_s>(),
        j.at( "newton_continuation" ).get<parameters_f::deflation_continuation_s::newton_extended_continuation_s>(),
        j.at( "newton_deflation" ).get<parameters_f::deflation_continuation_s::newton_extended_deflation_s>()
    };
}


void from_json( const nlohmann::json &j, parameters_d::stability_continuation_s::linear_solver_s &params_dc_lse_ )
{
    params_dc_lse_ = parameters_d::stability_continuation_s::linear_solver_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<double>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}
void from_json( const nlohmann::json &j, parameters_f::stability_continuation_s::linear_solver_s &params_dc_lse_ )
{
    params_dc_lse_ = parameters_f::stability_continuation_s::linear_solver_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<float>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}

void from_json( const nlohmann::json &j, parameters_d::stability_continuation_s::newton_s &params_dc_nt_ )
{
    params_dc_nt_ = parameters_d::stability_continuation_s::newton_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<double>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<double>()
    };
}
void from_json( const nlohmann::json &j, parameters_f::stability_continuation_s::newton_s &params_dc_nt_ )
{
    params_dc_nt_ = parameters_f::stability_continuation_s::newton_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<float>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<float>()
    };
}

void from_json( const nlohmann::json &j, parameters_d::stability_continuation_s &params_dc_ )
{
    params_dc_ = parameters_d::stability_continuation_s{
        j.at( "left_halfplane_stable_eigenvalues" ).get<bool>(),
        j.at( "Krylov_subspace_dimension" ).get<unsigned int>(),
        j.at( "desired_spectrum" ).get<unsigned int>(),
        j.at( "Cayley_transform_sigma_mu" ).get<std::vector<double>>(),
        j.at( "linear_solver" ).get<parameters_d::stability_continuation_s::linear_solver_s>(),
        j.at( "newton" ).get<parameters_d::stability_continuation_s::newton_s>()
    };
}
void from_json( const nlohmann::json &j, parameters_f::stability_continuation_s &params_dc_ )
{
    params_dc_ = parameters_f::stability_continuation_s{
        j.at( "left_halfplane_stable_eigenvalues" ).get<bool>(),
        j.at( "Krylov_subspace_dimension" ).get<unsigned int>(),
        j.at( "desired_spectrum" ).get<unsigned int>(),
        j.at( "Cayley_transform_sigma_mu" ).get<std::vector<float>>(),
        j.at( "linear_solver" ).get<parameters_f::stability_continuation_s::linear_solver_s>(),
        j.at( "newton" ).get<parameters_f::stability_continuation_s::newton_s>()
    };
}

void from_json( const nlohmann::json &j, parameters_d::nonlinear_operator_s::linear_solver_s &params_no_ls_ )
{
    // unsigned int lin_solver_max_it;
    // unsigned int use_precond_resid;
    // unsigned int resid_recalc_freq;
    // unsigned int basis_size;
    // T lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
    // bool save_convergence_history;
    // bool divide_out_norms_by_rel_base;
    // bool verbose;
    params_no_ls_ = parameters_d::nonlinear_operator_s::linear_solver_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<double>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}
void from_json( const nlohmann::json &j, parameters_f::nonlinear_operator_s::linear_solver_s &params_no_ls_ )
{
    params_no_ls_ = parameters_f::nonlinear_operator_s::linear_solver_s{
        j.at( "maximum_iterations" ).get<unsigned int>(),
        j.at( "use_preconditioned_residual" ).get<unsigned int>(),
        j.at( "residual_recalculate_frequency" ).get<unsigned int>(),
        j.at( "basis_size" ).get<unsigned int>(),
        j.at( "tolerance" ).get<float>(),
        j.at( "save_convergence_history" ).get<bool>(),
        j.at( "divide_norms_by_relative_base" ).get<bool>(),
        j.value( "verbose", true )
    };
}

void from_json( const nlohmann::json &j, parameters_d::nonlinear_operator_s::newton_s &params_no_nw_ )
{

    params_no_nw_ = parameters_d::nonlinear_operator_s::newton_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<double>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<double>()
    };
}
void from_json( const nlohmann::json &j, parameters_f::nonlinear_operator_s::newton_s &params_no_nw_ )
{

    params_no_nw_ = parameters_f::nonlinear_operator_s::newton_s{
        j.at( "maximum_iterations" ).get<unsigned int>(), j.at( "update_wight_maximum" ).get<float>(),
        j.at( "save_norms_history" ).get<bool>(), j.at( "verbose" ).get<bool>(), j.at( "tolerance" ).get<float>()
    };
}

void from_json( const nlohmann::json &j, parameters_d::nonlinear_operator_s &params_no_ )
{
    params_no_ = parameters_d::nonlinear_operator_s{
        j.at( "discrete_problem_dimensions" ).get<std::vector<size_t>>(),
        j.at( "problem_real_parameters_vector" ).get<std::vector<double>>(),
        j.at( "problem_int_parameters_vector" ).get<std::vector<int>>(),

        j.at( "linear_solver" ).get<parameters_d::nonlinear_operator_s::linear_solver_s>(),
        j.at( "newton" ).get<parameters_d::nonlinear_operator_s::newton_s>()

    };
}
void from_json( const nlohmann::json &j, parameters_f::nonlinear_operator_s &params_no_ )
{
    params_no_ = parameters_f::nonlinear_operator_s{
        j.at( "discrete_problem_dimensions" ).get<std::vector<size_t>>(),
        j.at( "problem_real_parameters_vector" ).get<std::vector<float>>(),
        j.at( "problem_int_parameters_vector" ).get<std::vector<int>>(),

        j.at( "linear_solver" ).get<parameters_f::nonlinear_operator_s::linear_solver_s>(),
        j.at( "newton" ).get<parameters_f::nonlinear_operator_s::newton_s>()

    };
}

void from_json( const nlohmann::json &j, parameters_d::plot_solutions_s &params_pl_ )
{
    params_pl_ = parameters_d::plot_solutions_s{ j.at( "plot_solution_frequency" ).get<int>() };
}
void from_json( const nlohmann::json &j, parameters_f::plot_solutions_s &params_pl_ )
{
    params_pl_ = parameters_f::plot_solutions_s{ j.at( "plot_solution_frequency" ).get<int>() };
}


void from_json( const nlohmann::json &j, parameters_d &params_ )
{
    params_ = parameters_d{
        j.at( "gpu_pci_id" ).get<int>(),
        j.at( "use_high_precision_reduction" ).get<bool>(),
        j.at( "path_to_project" ).get<std::string>(),
        j.at( "bifurcaiton_diagram_file_name" ).get<std::string>(),
        j.at( "stability_diagram_file_name" ).get<std::string>(),

        j.at( "deflation_continuation" ).get<parameters_d::deflation_continuation_s>(),
        j.at( "stability_continuation" ).get<parameters_d::stability_continuation_s>(),
        j.at( "nonlinear_operator" ).get<parameters_d::nonlinear_operator_s>(),
        j.at( "plot_solutions" ).get<parameters_d::plot_solutions_s>()

    };
}
void from_json( const nlohmann::json &j, parameters_f &params_ )
{
    params_ = parameters_f{
        j.at( "gpu_pci_id" ).get<int>(),
        j.at( "use_high_precision_reduction" ).get<bool>(),
        j.at( "path_to_project" ).get<std::string>(),
        j.at( "bifurcaiton_diagram_file_name" ).get<std::string>(),
        j.at( "stability_diagram_file_name" ).get<std::string>(),

        j.at( "deflation_continuation" ).get<parameters_f::deflation_continuation_s>(),
        j.at( "stability_continuation" ).get<parameters_f::stability_continuation_s>(),
        j.at( "nonlinear_operator" ).get<parameters_f::nonlinear_operator_s>(),
        j.at( "plot_solutions" ).get<parameters_f::plot_solutions_s>()

    };
}


nlohmann::json read_json( const std::string &project_file_name_ )
{
    try
    {
        std::ifstream f( project_file_name_ );
        if ( f )
        {
            nlohmann::json j;
            f >> j;
            return j;
        }
        else
        {
            throw std::runtime_error( std::string( "Failed to open file " ) + project_file_name_ + " for reading" );
        }
    }
    catch ( const nlohmann::json::exception &exception )
    {
        //std::cout << exception.what() << std::endl;
        std::throw_with_nested( std::runtime_error{ "json path: " + project_file_name_ + "\n" + exception.what() } );
    }
}


template <typename T>
parameters<T> read_parameters_json( const std::string &project_file_name_ )
{
    parameters<T> parameters_str;
    try
    {
        parameters_str = read_json( project_file_name_ ).get<parameters<T>>();
    }
    catch ( const std::exception &e )
    {
        std::throw_with_nested(
            std::runtime_error{
                "failed to read parameters JSON file: " + project_file_name_ + "\n" + e.what()
            });
    }
    return parameters_str;
}


}
#endif // __PARAMETERS_HPP__
