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
#include <fstream>

//includes json library by nlohmann
#include <contrib/json/nlohmann/json.hpp>

namespace main_classes
{

template<typename T>
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
            T lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
            bool is_small_alpha;   
            bool save_convergence_history;
            bool divide_out_norms_by_rel_base;


            void set_default()
            {
                lin_solver_max_it = 1500;
                use_precond_resid = 1;
                resid_recalc_freq = 1;
                basis_size = 4;
                lin_solver_tol = 5.0e-3;
                is_small_alpha = false;
                save_convergence_history = true;
                divide_out_norms_by_rel_base = true;
            }
        };
        struct newton_extended_continuation_s
        {
            unsigned int newton_max_it;
            T newton_wight;
            bool store_norms_history;
            bool verbose;            
            T tolerance;

            void set_default()
            {
                newton_max_it = 300;
                newton_wight = T(1.0);
                store_norms_history = true;
                verbose = true;            
                tolerance = 1.0e-9;                
            }
        };
        struct newton_extended_deflation_s
        {
            unsigned int newton_max_it;
            T newton_wight;
            bool store_norms_history;
            bool verbose;            
            T tolerance;
            void set_default()
            {
                newton_max_it = 300;
                newton_wight = T(0.5);
                store_norms_history = true;
                verbose = true;            
                tolerance = 1.0e-9;                
            }            
        };              
        
        unsigned int continuation_steps;
        T step_size; 
        unsigned int deflation_attempts;
        unsigned int continuation_fail_attempts;
        int initial_direciton;
        T step_ds_m;
        T step_ds_p;
        unsigned int skip_files;
        std::vector<T> deflation_knots;


        linear_solver_extended_s linear_solver_extended;
        newton_extended_continuation_s newton_extended_continuation;
        newton_extended_deflation_s newton_extended_deflation;

        void set_default()
        {
            continuation_steps = 5000;
            step_size = 0.5; 
            deflation_attempts = 5;
            continuation_fail_attempts = 4;
            initial_direciton = -1;
            step_ds_m = 0.2;
            step_ds_p = 0.01;
            skip_files = 100;
            deflation_knots = {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};    
            linear_solver_extended.set_default();
            newton_extended_continuation.set_default();  
            newton_extended_deflation.set_default();        
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
            
            void set_default()
            {
                lin_solver_max_it = 1500;
                use_precond_resid = 1;
                resid_recalc_freq = 1;
                basis_size = 4;
                lin_solver_tol = 5.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
                        //those are custom parameters to be set only to high dim Krylov methods
                save_convergence_history = true;
                divide_out_norms_by_rel_base = true;  
            }    
        };

        struct newton_s
        {
            unsigned int newton_max_it;
            T newton_wight;
            bool store_norms_history;
            bool verbose;            
            T tolerance;

            void set_default()
            {
                newton_max_it = 300;
                newton_wight = T(1.0);
                store_norms_history = true;
                verbose = true;            
                tolerance = 1.0e-9;                
            }
        };

        bool linear_operator_stable_eigenvalues_left_halfplane;
        unsigned int Krylov_subspace;

        linear_solver_s linear_solver;
        newton_s newton;  

        void set_default()
        {
            linear_operator_stable_eigenvalues_left_halfplane = true;
            Krylov_subspace = 15;
            linear_solver.set_default();
            newton.set_default();
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
            T lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
            bool save_convergence_history;
            bool divide_out_norms_by_rel_base;   

            void set_default()
            {
                lin_solver_max_it = 1500;
                use_precond_resid = 1;
                resid_recalc_freq = 1;
                basis_size = 4;
                lin_solver_tol = 5.0e-3; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
                save_convergence_history = true;
                divide_out_norms_by_rel_base = true;                   
            }         
        };

        struct newton_s
        {
            unsigned int newton_max_it;
            T newton_wight;
            bool store_norms_history;
            bool verbose;
            T tolerance;

            void set_default()
            {
                newton_max_it = 300;
                newton_wight = T(1.0);
                store_norms_history = true;
                verbose = true;
                tolerance = 1.0e-9;                
            }
        };

        std::vector<size_t> N_size;
        // problem dependent:
        std::vector< T > problem_real_parameters_vector;
        std::vector< int > problem_int_parameters_vector;
        
        linear_solver_s linear_solver;
        newton_s newton;

        void set_default()
        {
            N_size = {256, 256};
            
            problem_real_parameters_vector = {0.5, 2.0, 4.0};
            problem_int_parameters_vector = {2};

            linear_solver.set_default();
            newton.set_default();
        }
    };

    struct plot_solutions_s
    {
        int plot_solution_frequency;
        
        void set_default()
        {
            plot_solution_frequency = 3;
        }
    };

    int nvidia_pci_id;
    
    std::string path_to_prject; //relative to the execution root directory
    //for serialization, just the filenames
    std::string bifurcaiton_diagram_file_name;
    std::string stability_diagram_file_name;

    
    deflation_continuation_s deflation_continuation;
    stability_continuation_s stability_continuation;
    nonlinear_operator_s nonlinear_operator;
    plot_solutions_s plot_solutions;

    void set_default()
    {
        nvidia_pci_id = -1;
        path_to_prject = "../KS2D/";
        bifurcaiton_diagram_file_name = "bifurcation_diagram.dat";
        stability_diagram_file_name = "stability_diagram.dat";

        deflation_continuation.set_default();
        stability_continuation.set_default();
        nonlinear_operator.set_default();
        plot_solutions.set_default();
    }

    void plot_all()
    {

    }

};


typedef parameters<double> parameters_d;
typedef parameters<float> parameters_f;


void from_json(const nlohmann::json &j, parameters_d::deflation_continuation_s::linear_solver_extended_s &params_dc_lse_)
{
    params_dc_lse_ = parameters_d::deflation_continuation_s::linear_solver_extended_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<double>(),
        j.at("use_small_alpha_approximation").get<bool>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()
    };
}
void from_json(const nlohmann::json &j, parameters_f::deflation_continuation_s::linear_solver_extended_s &params_dc_lse_)
{
    params_dc_lse_ = parameters_f::deflation_continuation_s::linear_solver_extended_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<float>(),
        j.at("use_small_alpha_approximation").get<bool>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()
    };
}


void from_json(const nlohmann::json &j, parameters_d::deflation_continuation_s::newton_extended_continuation_s &params_dc_nec_)
{
    params_dc_nec_ = parameters_d::deflation_continuation_s::newton_extended_continuation_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<double>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<double>()
        
    };
}
void from_json(const nlohmann::json &j, parameters_f::deflation_continuation_s::newton_extended_continuation_s &params_dc_nec_)
{
    params_dc_nec_ = parameters_f::deflation_continuation_s::newton_extended_continuation_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<float>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<float>()
        
    };
}

void from_json(const nlohmann::json &j, parameters_d::deflation_continuation_s::newton_extended_deflation_s &params_dc_ned_)
{
    params_dc_ned_ = parameters_d::deflation_continuation_s::newton_extended_deflation_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<double>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<double>()
        
    };
}
void from_json(const nlohmann::json &j, parameters_f::deflation_continuation_s::newton_extended_deflation_s &params_dc_ned_)
{
    params_dc_ned_ = parameters_f::deflation_continuation_s::newton_extended_deflation_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<float>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<float>()
        
    };
}


void from_json(const nlohmann::json &j, parameters_d::deflation_continuation_s &params_dc_)
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
    params_dc_ = parameters_d::deflation_continuation_s
    {
        j.at("continuation_steps").get<unsigned int>(),
        j.at("step_size").get<double>(),
        j.at("deflation_attempts").get<unsigned int>(),
        j.at("continuation_fail_attempts").get<unsigned int>(),
        j.at("initial_direciton").get<int>(),
        j.at("minimum_step_multiplier").get<double>(),
        j.at("maximum_step_multiplier").get<double>(),
        j.at("skip_file_output").get<unsigned int>(),
        j.at("deflation_knots").get< std::vector<double> >(),

        j.at("linear_solver_extended").get< parameters_d::deflation_continuation_s::linear_solver_extended_s >(),
        j.at("newton_continuation").get< parameters_d::deflation_continuation_s::newton_extended_continuation_s >(),
        j.at("newton_deflation").get< parameters_d::deflation_continuation_s::newton_extended_deflation_s >() 
    };
}
void from_json(const nlohmann::json &j, parameters_f::deflation_continuation_s &params_dc_)
{  
    params_dc_ = parameters_f::deflation_continuation_s
    {
        j.at("continuation_steps").get<unsigned int>(),
        j.at("step_size").get<float>(),
        j.at("deflation_attempts").get<unsigned int>(),
        j.at("continuation_fail_attempts").get<unsigned int>(),
        j.at("initial_direciton").get<int>(),
        j.at("minimum_step_multiplier").get<float>(),
        j.at("maximum_step_multiplier").get<float>(),
        j.at("skip_file_output").get<unsigned int>(),
        j.at("deflation_knots").get< std::vector<float> >(),

        j.at("linear_solver_extended").get< parameters_f::deflation_continuation_s::linear_solver_extended_s >(),
        j.at("newton_continuation").get< parameters_f::deflation_continuation_s::newton_extended_continuation_s >(),
        j.at("newton_deflation").get< parameters_f::deflation_continuation_s::newton_extended_deflation_s >()    
    };
}


void from_json(const nlohmann::json &j, parameters_d::stability_continuation_s::linear_solver_s &params_dc_lse_)
{
    params_dc_lse_ = parameters_d::stability_continuation_s::linear_solver_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<double>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()
    };
}
void from_json(const nlohmann::json &j, parameters_f::stability_continuation_s::linear_solver_s &params_dc_lse_)
{
    params_dc_lse_ = parameters_f::stability_continuation_s::linear_solver_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<float>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()
    };
}

void from_json(const nlohmann::json &j, parameters_d::stability_continuation_s::newton_s &params_dc_nt_)
{
    params_dc_nt_ = parameters_d::stability_continuation_s::newton_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<double>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<double>() 
    };
}
void from_json(const nlohmann::json &j, parameters_f::stability_continuation_s::newton_s &params_dc_nt_)
{
    params_dc_nt_ = parameters_f::stability_continuation_s::newton_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<float>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<float>() 
    };
}

void from_json(const nlohmann::json &j, parameters_d::stability_continuation_s &params_dc_)
{
    params_dc_ = parameters_d::stability_continuation_s
    {
        j.at("left_halfplane_stable_eigenvalues").get<bool>(),
        j.at("Krylov_subspace_dimension").get<unsigned int>(),

        j.at("linear_solver").get< parameters_d::stability_continuation_s::linear_solver_s >(),
        j.at("newton").get< parameters_d::stability_continuation_s::newton_s >()
    };
}
void from_json(const nlohmann::json &j, parameters_f::stability_continuation_s &params_dc_)
{
    params_dc_ = parameters_f::stability_continuation_s
    {
        j.at("left_halfplane_stable_eigenvalues").get<bool>(),
        j.at("Krylov_subspace_dimension").get<unsigned int>(),
        
        j.at("linear_solver").get< parameters_f::stability_continuation_s::linear_solver_s >(),
        j.at("newton").get< parameters_f::stability_continuation_s::newton_s >()        
    };
}

void from_json(const nlohmann::json &j, parameters_d::nonlinear_operator_s::linear_solver_s &params_no_ls_)
{
    // unsigned int lin_solver_max_it;
    // unsigned int use_precond_resid;
    // unsigned int resid_recalc_freq;
    // unsigned int basis_size;
    // T lin_solver_tol; //relative tolerance wrt to rhs vector. For Krylov-Newton method can be set lower
    // bool save_convergence_history;
    // bool divide_out_norms_by_rel_base;   
    params_no_ls_ = parameters_d::nonlinear_operator_s::linear_solver_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<double>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()        
    };
}
void from_json(const nlohmann::json &j, parameters_f::nonlinear_operator_s::linear_solver_s &params_no_ls_)
{  
    params_no_ls_ = parameters_f::nonlinear_operator_s::linear_solver_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("use_preconditioned_residual").get<unsigned int>(),
        j.at("residual_recalculate_frequency").get<unsigned int>(),
        j.at("basis_size").get<unsigned int>(),
        j.at("tolerance").get<float>(),
        j.at("save_convergence_history").get<bool>(),
        j.at("divide_norms_by_relative_base").get<bool>()        
    };
}

void from_json(const nlohmann::json &j, parameters_d::nonlinear_operator_s::newton_s &params_no_nw_)
{

    params_no_nw_ = parameters_d::nonlinear_operator_s::newton_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<double>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<double>()
    };
}
void from_json(const nlohmann::json &j, parameters_f::nonlinear_operator_s::newton_s &params_no_nw_)
{

    params_no_nw_ = parameters_f::nonlinear_operator_s::newton_s
    {
        j.at("maximum_iterations").get<unsigned int>(),
        j.at("update_wight_maximum").get<float>(),
        j.at("save_norms_history").get<bool>(),
        j.at("verbose").get<bool>(),
        j.at("tolerance").get<float>() 
    };
}

void from_json(const nlohmann::json &j, parameters_d::nonlinear_operator_s &params_no_)
{
    params_no_ = parameters_d::nonlinear_operator_s
    {
        j.at("discrete_problem_dimensions").get< std::vector<size_t> >(),
        j.at("problem_real_parameters_vector").get< std::vector< double > >(),
        j.at("problem_int_parameters_vector").get< std::vector< int > >(),

        j.at("linear_solver").get< parameters_d::nonlinear_operator_s::linear_solver_s >(),
        j.at("newton").get< parameters_d::nonlinear_operator_s::newton_s >()

    };
}
void from_json(const nlohmann::json &j, parameters_f::nonlinear_operator_s &params_no_)
{
    params_no_ = parameters_f::nonlinear_operator_s
    {
        j.at("discrete_problem_dimensions").get< std::vector<size_t> >(),
        j.at("problem_real_parameters_vector").get< std::vector< float > >(),
        j.at("problem_int_parameters_vector").get< std::vector< int > >(),

        j.at("linear_solver").get< parameters_f::nonlinear_operator_s::linear_solver_s >(),
        j.at("newton").get< parameters_f::nonlinear_operator_s::newton_s >()        

    };
}

void from_json(const nlohmann::json &j, parameters_d::plot_solutions_s &params_pl_)
{
    params_pl_ = parameters_d::plot_solutions_s
    {
        j.at("plot_solution_frequency").get< int >()
    };
}
void from_json(const nlohmann::json &j, parameters_f::plot_solutions_s &params_pl_)
{
    params_pl_ = parameters_f::plot_solutions_s
    {
        j.at("plot_solution_frequency").get< int >()
    };
}


void from_json(const nlohmann::json &j, parameters_d &params_)
{
    params_ = parameters_d
    {
        j.at("gpu_pci_id").get< int >(),
        j.at("path_to_prject").get< std::string >(),
        j.at("bifurcaiton_diagram_file_name").get< std::string >(),
        j.at("stability_diagram_file_name").get< std::string >(),
        
        j.at("deflation_continuation").get< parameters_d::deflation_continuation_s >(),
        j.at("stability_continuation").get< parameters_d::stability_continuation_s >(),
        j.at("nonlinear_operator").get< parameters_d::nonlinear_operator_s >(),
        j.at("plot_solutions").get< parameters_d::plot_solutions_s >()
        
    };
}
void from_json(const nlohmann::json &j, parameters_f &params_)
{
    params_ = parameters_f
    {
        j.at("gpu_pci_id").get< int >(),
        j.at("path_to_prject").get< std::string >(),
        j.at("bifurcaiton_diagram_file_name").get< std::string >(),
        j.at("stability_diagram_file_name").get< std::string >(),
        
        j.at("deflation_continuation").get< parameters_f::deflation_continuation_s >(),
        j.at("stability_continuation").get< parameters_f::stability_continuation_s >(),
        j.at("nonlinear_operator").get< parameters_f::nonlinear_operator_s >(),
        j.at("plot_solutions").get< parameters_f::plot_solutions_s >()
        
    };
}


nlohmann::json read_json(const std::string &project_file_name_)
{
    try
    {
        std::ifstream f(project_file_name_);
        if (f)
        {
            nlohmann::json j;
            f >> j;
            return j;
        }
        else
        {
            throw std::runtime_error(std::string("Failed to open file ") + project_file_name_ + " for reading");
        }
    }
    catch (const nlohmann::json::exception &exception)
    {
        std::throw_with_nested(std::runtime_error{"json path: " + project_file_name_});
    }
}


template<typename T>
parameters<T> read_parameters_json(const std::string &project_file_name_)
{
    parameters<T> parameters_str;
    try
    {
        parameters_str = read_json(project_file_name_).get< parameters<T> >();
    }
    catch(const std::exception& e)
    {
        std::cout << "====================X====================" <<std::endl;
        std::cout << e.what() << std::endl << "setting default values for parameters structure" << std::endl;
        std::cout << "====================X====================" <<std::endl;
        parameters_str.set_default();
    }
    return parameters_str;
}


}
#endif // __PARAMETERS_HPP__