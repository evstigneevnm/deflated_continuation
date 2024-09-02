#ifndef __CONTINUATION_SINGLE_FILE_JSON_PARAMS_HPP__
#define __CONTINUATION_SINGLE_FILE_JSON_PARAMS_HPP__

#include <contrib/json/nlohmann/json.hpp>
#include <string>


namespace file_params
{

std::string __print_bool(bool val_)
{
    return( val_?"true":"false" );
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
        std::throw_with_nested(std::runtime_error{"json path: " + project_file_name_ + "\n" + exception.what()});
    }
}

struct parameters_s
{
    int steps;
    int skip_output;
    double R_min;
    double R_max;
    double alpha;
    size_t N;
    int high_prec;
    double dS_0;
    double dSmax;
    double dSdec_mult;
    double dSinc_mult;
    struct solvers_config_s
    {
        struct linear_solver_s
        {   
            unsigned int max_iterations;
            double tolerance;
            unsigned int use_precond_resid;
            unsigned int resid_recalc_freq;
            unsigned int basis_sz;
            void plot_all()
            {
                std::cout << "  linear_solver:" << std::endl;
                std::cout << "      max_iterations: " << max_iterations << std::endl;
                std::cout << "      tolerance: " << tolerance << std::endl;
                std::cout << "      use_precond_resid: " << use_precond_resid << std::endl;
                std::cout << "      resid_recalc_freq: " << resid_recalc_freq << std::endl;
                std::cout << "      basis_sz: " << basis_sz << std::endl;

            }
        };
        struct linear_solver_extended_s
        {   
            unsigned int max_iterations;
            double tolerance;
            unsigned int use_precond_resid;
            unsigned int resid_recalc_freq;
            unsigned int basis_sz;
            void plot_all()
            {
                std::cout << "  linear_solver_extended:" << std::endl;
                std::cout << "      max_iterations: " << max_iterations << std::endl;
                std::cout << "      tolerance: " << tolerance << std::endl;
                std::cout << "      use_precond_resid: " << use_precond_resid << std::endl;
                std::cout << "      resid_recalc_freq: " << resid_recalc_freq << std::endl;
                std::cout << "      basis_sz: " << basis_sz << std::endl;

            }            
        };  
        struct newton_method_s
        {
            unsigned int max_iterations;
            double tolerance;
            void plot_all()
            {
                std::cout << "  newton_method:" << std::endl;
                std::cout << "      max_iterations: " << max_iterations << std::endl;
                std::cout << "      tolerance: " << tolerance << std::endl;

            }             
        };   
        struct newton_method_extended_s
        {
            unsigned int max_iterations;
            double tolerance;
            double update_wight_maximum;
            bool save_norms_history;
            bool verbose;
            double relax_tolerance_factor;
            unsigned int relax_tolerance_steps;
            unsigned int stagnation_p_max;
            double maximum_norm_increase;
            void plot_all()
            {
                std::cout << "  newton_method_extended:" << std::endl;
                std::cout << "      max_iterations: " << max_iterations << std::endl;
                std::cout << "      update_wight_maximum: " << update_wight_maximum << std::endl;
                std::cout << "      save_norms_history: " << __print_bool(save_norms_history) << std::endl;
                std::cout << "      verbose: " << __print_bool(verbose) << std::endl;
                std::cout << "      relax_tolerance_factor: " << relax_tolerance_factor << std::endl;
                std::cout << "      relax_tolerance_steps: " << relax_tolerance_steps << std::endl;
                std::cout << "      stagnation_p_max: " << stagnation_p_max << std::endl;
                std::cout << "      maximum_norm_increase: " << maximum_norm_increase << std::endl;

            }             
            
        };

        linear_solver_s linear_solver;
        linear_solver_extended_s linear_solver_extended;
        newton_method_s newton_method;
        newton_method_extended_s newton_method_extended;

        void plot_all()
        {
            std::cout << "solvers_config:" << std::endl;
            linear_solver.plot_all();
            linear_solver_extended.plot_all();
            newton_method.plot_all();
            newton_method_extended.plot_all();
        }


    };
    solvers_config_s solvers_config;

    void plot_all()
    {
        std::cout << std::endl;
        std::cout << "steps: " << steps << std::endl;
        std::cout << "skip_output: " << skip_output << std::endl;
        std::cout << "R_min: " << R_min << std::endl;
        std::cout << "R_max: " << R_max << std::endl;

        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "N: " << N << std::endl;
        std::cout << "high_prec: " << high_prec << std::endl;
        std::cout << "dS_0: " << dS_0 << std::endl;
        std::cout << "dSmax: " << dSmax << std::endl;
        std::cout << "dSdec_mult: " << dSdec_mult << std::endl;
        std::cout << "dSinc_mult: " << dSinc_mult << std::endl;
        solvers_config.plot_all();

    }

};

parameters_s parameters;


void from_json(const nlohmann::json &j, parameters_s::solvers_config_s::linear_solver_s &params_ls)
{
    params_ls = parameters_s::solvers_config_s::linear_solver_s
    {
        j.at("max_iterations").get< unsigned int >(),
        j.at("tolerance").get< double >(),
        j.at("use_precond_resid").get< unsigned int >(),
        j.at("resid_recalc_freq").get< unsigned int >(),
        j.at("basis_sz").get< unsigned int >()

    };
}

void from_json(const nlohmann::json &j, parameters_s::solvers_config_s::linear_solver_extended_s &params_ls)
{
    params_ls = parameters_s::solvers_config_s::linear_solver_extended_s
    {
        j.at("max_iterations").get< unsigned int >(),
        j.at("tolerance").get< double >(),
        j.at("use_precond_resid").get< unsigned int >(),
        j.at("resid_recalc_freq").get< unsigned int >(),
        j.at("basis_sz").get< unsigned int >()

    };
}

void from_json(const nlohmann::json &j, parameters_s::solvers_config_s::newton_method_s &params_ls)
{
    params_ls = parameters_s::solvers_config_s::newton_method_s
    {
        j.at("max_iterations").get< unsigned int >(),
        j.at("tolerance").get< double >()
    };
}

void from_json(const nlohmann::json &j, parameters_s::solvers_config_s::newton_method_extended_s &params_ls)
{
    params_ls = parameters_s::solvers_config_s::newton_method_extended_s
    {
        j.at("max_iterations").get< unsigned int >(),
        j.at("tolerance").get< double >(),
        j.at("update_wight_maximum").get< double >(),
        j.at("save_norms_history").get< bool >(),
        j.at("verbose").get< bool >(),
        j.at("relax_tolerance_factor").get< double >(),
        j.at("relax_tolerance_steps").get< unsigned int >(),
        j.at("stagnation_p_max").get< unsigned int >(),
        j.at("maximum_norm_increase").get< double >()
    };
}

void from_json(const nlohmann::json &j, parameters_s::solvers_config_s &params_solvers_config_)
{
    params_solvers_config_ = parameters_s::solvers_config_s
    {
        j.at("linear_solver").get< parameters_s::solvers_config_s::linear_solver_s >(),
        j.at("linear_solver_extended").get< parameters_s::solvers_config_s::linear_solver_extended_s >(),
        j.at("newton_method").get< parameters_s::solvers_config_s::newton_method_s >(),
        j.at("newton_method_extended").get< parameters_s::solvers_config_s::newton_method_extended_s >()
    };
}


void from_json(const nlohmann::json &j, parameters_s &params_)
{
    params_ = parameters_s
    {
        j.at("steps").get< int >(),        
        j.at("skip_output").get< int >(),
        j.at("R_min").get< double >(),
        j.at("R_max").get< double >(),
        j.at("alpha").get< double >(),
        j.at("N").get< size_t >(),
        j.at("high_prec").get< int >(),
        j.at("dS_0").get< double >(),
        j.at("dSmax").get< double >(),
        j.at("dSdec_mult").get< double >(),
        j.at("dSinc_mult").get< double >(),

        j.at("solvers_config").get< parameters_s::solvers_config_s >()     
    };
}


parameters_s read_parameters_json(const std::string &project_file_name_)
{
    parameters_s parameters_str;
    try
    {
        parameters_str = read_json(project_file_name_).get< parameters_s >();
    }
    catch(const std::exception& e)
    {
        std::cout << "====================X====================" <<std::endl;
        std::cout << "failed to read json file because:" << std::endl;
        std::cout << e.what() << std::endl;
        std::cout << "====================X====================" <<std::endl;
        throw e;
    }
    parameters_str.plot_all();
    return parameters_str;
}



}

#endif