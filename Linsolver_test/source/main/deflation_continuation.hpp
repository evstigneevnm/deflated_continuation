#ifndef __DEFLATION_CONTINUATION_HPP__
#define __DEFLATION_CONTINUATION_HPP__

/**
*   The main class of the whole Deflation-Continuaiton Process (DCP). 
*
*   It uses nonlinear operator and other set options to configure the whole project.
*   After vector and file operations, the nonlinear operator, log and monitor are configured,
*   this class is initialized and configured to perform the whole DCP.
*   data serialization is done using boost archive
*/
#include <string>
#include <vector>
// #include <type_traits> //to check linsolvers
//boost serializatoin
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>


#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <containers/knots.hpp>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>

#include <continuation/continuation.hpp>
#include <continuation/continuation_analytical.hpp> // inherited from continuation to put a nontrivial analytical solution on the curve, if needed.

#include <deflation/solution_storage.h>
#include <deflation/deflation.hpp>



namespace main_classes{


template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator, class Parameters>
class deflation_continuation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;

    typedef typename boost::archive::text_oarchive data_output;
    typedef typename boost::archive::text_iarchive data_input;

    //general linear solver used in continuation and deflation
    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        LinearOperator,
        Preconditioner,
        VectorOperations,
        monitor_t,
        Log,
        LinearSolver
        > sherman_morrison_linear_system_solve_t;

    //general system operator for the newton's method
    //TODO: move to nonlinear_operators?
    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;

    //convergence strategy for the newton's method for F(x) = 0
    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;
    //newton's method for F(x) = 0
    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;
    
    typedef container::knots<T> knots_t;

    typedef container::curve_helper_container<VectorOperations> container_helper_t;

    typedef deflation::solution_storage<VectorOperations> sol_storage_def_t;   


    typedef container::bifurcation_diagram_curve<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        container_helper_t
        > bif_diag_curve_t;

    typedef container::bifurcation_diagram<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        bif_diag_curve_t,
        container_helper_t
        > bif_diag_t;

    typedef continuation::continuation<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t
        > continuate_t;

    typedef continuation::continuation_analytical<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t
        > continuate_analytical_t;

    typedef deflation::deflation<
        VectorOperations,
        VectorFileOperations,
        Log,
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t
        > deflate_t;


public:
    deflation_continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, Parameters* parameters_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    log_linsolver(log_linsolver_),
    parameters(parameters_)
    {
        
        //add '/' to the end of the project dir, if needed
        
        project_dir = parameters->path_to_prject;
        skip_files = parameters->deflation_continuation.skip_files;

        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';


        lin_op = new LinearOperator(nonlin_op);
        precond = new Preconditioner(nonlin_op);
        SM = new sherman_morrison_linear_system_solve_t(precond, vec_ops, log_linsolver);
        conv_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, SM);
        newton = new newton_t(vec_ops, system_operator, conv_newton);
        knots = new knots_t();
        continuate = new continuate_t(vec_ops, file_ops, log, nonlin_op, lin_op, knots, SM, newton);
        continuate_analytical = new continuate_analytical_t(vec_ops, file_ops, log, nonlin_op, lin_op, knots, SM, newton);
        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files);
        sol_storage_def = new sol_storage_def_t(vec_ops, 50, vec_ops->get_l2_size(), 2.0 );  //T(1.0) is a norm_wight! Used as sqrt(N) for L2 norm. Use it again? Check this!!!
        deflate = new deflate_t(vec_ops, file_ops, log, nonlin_op, lin_op, SM, sol_storage_def);

    }
    ~deflation_continuation()
    {
        
        delete deflate;
        delete sol_storage_def;
        delete bif_diag;
        delete continuate;
        delete continuate_analytical;
        delete knots;
        delete newton;
        delete system_operator;
        delete conv_newton;
        delete SM;
        delete precond;
        delete lin_op;
    }



//  called to set all parameters from the parameter structure
    void set_parameters()
    {
        set_linsolver();
        set_extended_linsolver();
        set_newton();
        set_newton_continuation();
        set_newton_deflation();
        set_steps();
        set_deflation_knots();
    }

    void set_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true*/
    {
        //setup linear system:
        mon_orig = &SM->get_linsolver_handle_original()->monitor();
        T lin_solver_tol = parameters->nonlinear_operator.linear_solver.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->nonlinear_operator.linear_solver.lin_solver_max_it;
        bool save_convergence_history_ = parameters->nonlinear_operator.linear_solver.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->nonlinear_operator.linear_solver.divide_out_norms_by_rel_base;
        int use_precond_resid = parameters->nonlinear_operator.linear_solver.use_precond_resid;
        int resid_recalc_freq = parameters->nonlinear_operator.linear_solver.resid_recalc_freq;
        int basis_sz = parameters->nonlinear_operator.linear_solver.basis_size;

        mon_orig->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon_orig->set_save_convergence_history(save_convergence_history_);
        mon_orig->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon_orig->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            SM->get_linsolver_handle_original()->set_basis_size(basis_sz);  
            // SM->get_linsolver_handle_original()->set_restarts(basis_sz);  
//
    }
    void set_extended_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, bool is_small_alpha = false, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true
*/    
    {
        mon = &SM->get_linsolver_handle()->monitor();
        T lin_solver_tol = parameters->deflation_continuation.linear_solver_extended.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->deflation_continuation.linear_solver_extended.lin_solver_max_it;
        bool save_convergence_history_ = parameters->deflation_continuation.linear_solver_extended.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->deflation_continuation.linear_solver_extended.divide_out_norms_by_rel_base;
        int use_precond_resid = parameters->deflation_continuation.linear_solver_extended.use_precond_resid;
        int resid_recalc_freq = parameters->deflation_continuation.linear_solver_extended.resid_recalc_freq;
        int basis_sz = parameters->deflation_continuation.linear_solver_extended.basis_size;
        bool is_small_alpha = parameters->deflation_continuation.linear_solver_extended.is_small_alpha;

        mon->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon->set_save_convergence_history(save_convergence_history_);
        mon->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
           SM->get_linsolver_handle()->set_basis_size(basis_sz); 
            // SM->get_linsolver_handle()->set_restarts(basis_sz); 
//
        SM->is_small_alpha(is_small_alpha);        
    }
    void set_newton()
    /*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/
    {
        T tolerance_ = parameters->nonlinear_operator.newton.tolerance;
        unsigned int maximum_iterations_ = parameters->nonlinear_operator.newton.newton_max_it;
        T newton_wight_ = parameters->nonlinear_operator.newton.newton_wight;

        bool store_norms_history_ = parameters->nonlinear_operator.newton.store_norms_history;
        bool verbose_ = parameters->nonlinear_operator.newton.verbose;

        conv_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);
    }

    void set_newton_continuation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.8), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        T tolerance_ = parameters->deflation_continuation.newton_extended_continuation.tolerance;
        unsigned int maximum_iterations_ = parameters->deflation_continuation.newton_extended_continuation.newton_max_it;
        T newton_wight_ = parameters->deflation_continuation.newton_extended_continuation.newton_wight;

        bool store_norms_history_ = parameters->deflation_continuation.newton_extended_continuation.store_norms_history;
        bool verbose_ = parameters->deflation_continuation.newton_extended_continuation.verbose;

        continuate->set_newton(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);

    }

    void set_newton_deflation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        
        T tolerance_ = parameters->deflation_continuation.newton_extended_deflation.tolerance;
        unsigned int maximum_iterations_ = parameters->deflation_continuation.newton_extended_deflation.newton_max_it;
        T newton_wight_ = parameters->deflation_continuation.newton_extended_deflation.newton_wight;

        bool store_norms_history_ = parameters->deflation_continuation.newton_extended_deflation.store_norms_history;
        bool verbose_ = parameters->deflation_continuation.newton_extended_deflation.verbose;
        deflate->set_newton(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);

    }

    void set_steps()
/*unsigned int max_S_, T ds_0_, unsigned int deflation_attempts_ = 5, unsigned int attempts_0_ = 4, int initial_direciton_ = -1, T step_ds_m_ = 0.2, T step_ds_p_ = 0.01*/    
    {
        
        unsigned int max_S_ = parameters->deflation_continuation.continuation_steps;
        T ds_0_ = parameters->deflation_continuation.step_size;
        int initial_direciton_ = parameters->deflation_continuation.initial_direciton;
        T step_ds_m_ = parameters->deflation_continuation.step_ds_m;
        T step_ds_p_ = parameters->deflation_continuation.step_ds_p;
        unsigned int attempts_0_ = parameters->deflation_continuation.continuation_fail_attempts;
        unsigned int deflation_attempts_ = parameters->deflation_continuation.deflation_attempts;


        deflate->set_max_retries(deflation_attempts_);
        continuate->set_steps(max_S_, ds_0_, initial_direciton_, step_ds_m_, step_ds_p_, attempts_0_);
        continuate_analytical->set_steps(max_S_, ds_0_, initial_direciton_, step_ds_m_, step_ds_p_, attempts_0_);
    }

    void set_deflation_knots()
/*std::vector<T> knots_*/    
    {
        knots->add_element(parameters->deflation_continuation.deflation_knots);
    }

    void use_analytical_solution(bool analytical_solution_ = false)
    {
        analytical_solution = analytical_solution_;
    }


    bool load_data(const std::string& file_name_ = {})
    {
        bool file_exists = false;
        if(!file_name_.empty())
        {
            std::ifstream load_file( (project_dir + file_name_).c_str() );
            if(load_file.good())
            {
                log->info_f("MAIN:deflation_continuation: reading data for the bifurcaiton diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                ia >> (*bif_diag);
                load_file.close();
                log->info_f("MAIN:deflation_continuation: read data for the bifurcaiton diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:deflation_continuation: failed to load saved data for the bifurcaiton diagram %s", (project_dir + file_name_).c_str() );
                file_exists = false;
            }
        }
        return file_exists;
    }

    void save_data(const std::string& file_name_ = {})
    {
        if(!file_name_.empty())
        {
            log->info_f("MAIN:deflation_continuation: saving data for the bifurcaiton diagram in %s ...", (project_dir + file_name_).c_str() );
            std::ofstream save_file( (project_dir + file_name_).c_str() );
            data_output oa(save_file);
            oa << (*bif_diag);
            save_file.close();
            log->info_f("MAIN:deflation_continuation: saved data for the bifurcaiton diagram in %s", (project_dir + file_name_).c_str() );
        }        
    }

    void edit()
    {
        std::string file_name = parameters->bifurcaiton_diagram_file_name;
        bool file_exists = load_data( file_name );
        if(file_exists)
        {
            std::cout << "entering interactive edit mode" << std::endl;
            std::cout << "enter 'd' to pop_back() the curve or 'q' to quit." << std::endl;
            char c = 'c';
            while(c != 'q')
            {
                std::cout << "file " << file_name << " contains:" << std::endl;
                bif_diag->print_curves_status();
                c = std::cin.get();
                if(c=='d')
                {
                    bif_diag->pop_back_curve();
                }
            }
            c = std::cin.get();
            std::cout << "save file(y/n)>>>";
            c = std::cin.get();
            if(c == 'y')
                save_data(file_name);

        }
        else
        {
            log->warning_f("MAIN:deflation_continuation: file %s doesn't exist; called edit with no file provided!", file_name.c_str());
        }

    }


    void execute()
    {
        std::string file_name = parameters->bifurcaiton_diagram_file_name;
        //Algorithm pseudocode:
        //
        //Set second knot value: knots.next()
        //while(true)
        //{
        //  knot_value = knot.get_value()
        //  perform deflation until a new solutoin is found
        //  if the solution is found:
        //    create a new curve and continuate it until it is finished
        //    add interseciton of the curve with the current knot value to the deflator: deflation_container.push_back()
        //  else 
        //    if(!knot.next())
        //      break
        //    else
        //      deflation_container.clear()
        //}
        //

        bool file_exists = load_data(file_name);

        // force file printing skip after serialization
        // unless it is done, the skip data is taken from the serialization class from file!
        bif_diag->set_skip_output(skip_files);


        T_vec x_deflation; //pointer to the found deflated solution
        bool is_there_a_next_knot = knots->next();

        //perform analytical solution continuation if desired and if it is the first run
        if( (analytical_solution)&&(!file_exists))
        {
            log->info("MAIN:deflation_continuation: using the analytical solution to form a curve...");
            bif_diag_curve_t* bdf;
            T lambda = knots->get_value();
            vec_ops->init_vector(x_deflation); vec_ops->start_use_vector(x_deflation);
            nonlin_op->exact_solution(lambda, x_deflation);
            bif_diag->init_new_curve();
            bif_diag->get_current_ref(bdf);
            continuate_analytical->continuate_curve(bdf, x_deflation, lambda);
            bif_diag->close_curve();
            save_data(file_name);
            vec_ops->stop_use_vector(x_deflation); vec_ops->free_vector(x_deflation);
            log->info("MAIN:deflation_continuation: analytical solution formed.");
        }
        //

        int number_of_solutions = 0;
        while(is_there_a_next_knot)
        {
            
            T lambda = knots->get_value();
            sol_storage_def->clear();
            log->info_f("MAIN:deflation_continuation: currently having %i curves.", bif_diag->current_curve() );

            bif_diag->find_intersection(lambda, sol_storage_def);

            bool is_new_solution = deflate->find_solution(lambda);
            if(is_new_solution)
            {
                number_of_solutions++;
                log->info_f("MAIN:deflation_continuation: found %i solutions for lambda = %lf.", number_of_solutions, double(lambda) );
                
                deflate->get_solution_ref(x_deflation);
                bif_diag_curve_t* bdf;
                
                bif_diag->init_new_curve();
                bif_diag->get_current_ref(bdf);
                //std::cin.get(); 
                continuate->continuate_curve(bdf, x_deflation, lambda);
                bif_diag->close_curve();
                //std::cin.get(); 
                save_data(file_name);
                
                bdf->find_intersection(lambda, sol_storage_def);
            }
            else
            {
                is_there_a_next_knot = knots->next();
            }
        }
    }



private:
//  references to the external classes:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    Parameters* parameters;

//created locally:
    LinearOperator* lin_op;
    Preconditioner* precond;
    monitor_t* mon;
    monitor_t* mon_orig; 
    sherman_morrison_linear_system_solve_t* SM;
    convergence_newton_t* conv_newton;
    system_operator_t* system_operator;
    newton_t* newton;
    knots_t* knots;
    bif_diag_t* bif_diag = nullptr;
    continuate_t* continuate;
    continuate_analytical_t* continuate_analytical;
    deflate_t* deflate;
    sol_storage_def_t* sol_storage_def;
    std::string project_dir;
    bool analytical_solution = false;
    unsigned int skip_files;
};


}

#endif // __DEFLATION_CONTINUATION_HPP__