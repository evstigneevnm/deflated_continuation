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

#include <deflation/solution_storage.h>
#include <deflation/deflation.hpp>



namespace main_classes{


template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator>
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
    deflation_continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, NonlinearOperations* nonlin_op_, const std::string& project_dir_, unsigned int skip_files_ = 10):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    project_dir(project_dir_)
    {
        
        //add '/' to the end of the project dir, if needed
        
        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';


        lin_op = new LinearOperator(nonlin_op);
        precond = new Preconditioner(nonlin_op);
        SM = new sherman_morrison_linear_system_solve_t(precond, vec_ops, log);
        conv_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, SM);
        newton = new newton_t(vec_ops, system_operator, conv_newton);
        knots = new knots_t();
        continuate = new continuate_t(vec_ops, file_ops, log, nonlin_op, lin_op, knots, SM, newton);
        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files_);
        sol_storage_def = new sol_storage_def_t(vec_ops, 50, vec_ops->get_l2_size(), 1.5 );  //T(1.0) is a norm_wight! Used as sqrt(N) for L2 norm. Use it again? Check this!!!
        deflate = new deflate_t(vec_ops, file_ops, log, nonlin_op, lin_op, SM, sol_storage_def);
    }
    ~deflation_continuation()
    {
        
        delete deflate;
        delete sol_storage_def;
        delete bif_diag;
        delete continuate;
        delete knots;
        delete newton;
        delete system_operator;
        delete conv_newton;
        delete SM;
        delete precond;
        delete lin_op;
    }



//TODO: change all this rubbish to a single class that populates the structure. 
//      It can be populated form anywhere: stdin, file, pipe etc.

    void set_linsolver(T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4)
    {
        //setup linear system:
        mon_orig = &SM->get_linsolver_handle_original()->monitor();
        mon_orig->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon_orig->set_save_convergence_history(true);
        mon_orig->set_divide_out_norms_by_rel_base(true);
        mon_orig->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            SM->get_linsolver_handle_original()->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            SM->get_linsolver_handle_original()->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            SM->get_linsolver_handle_original()->set_basis_size(basis_sz);  
//
    }
    void set_extended_linsolver(T lin_solver_tol, unsigned int lin_solver_max_it, bool is_small_alpha = false, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4)
    {
        mon = &SM->get_linsolver_handle()->monitor();
        mon->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon->set_save_convergence_history(true);
        mon->set_divide_out_norms_by_rel_base(true);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            SM->get_linsolver_handle()->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            SM->get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            SM->get_linsolver_handle()->set_basis_size(basis_sz); 
//
        SM->is_small_alpha(is_small_alpha);        
    }
    void set_newton(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true)
    {
        conv_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);
        continuate->set_newton(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);
        deflate->set_newton(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);

    }
    void set_steps(unsigned int max_S_, T ds_0_, unsigned int deflation_attempts_ = 5, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        deflate->set_max_retries(deflation_attempts_);
        continuate->set_steps(max_S_, ds_0_, initial_direciton_, step_ds_m_, step_ds_p_, attempts_0_);
    }

    void set_deflation_knots(std::vector<T> knots_)
    {
        knots->add_element(knots_);
    }


    void load_data(const std::string& file_name_ = {})
    {
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
            }
            else
            {
                log->warning_f("MAIN:deflation_continuation: failed to load saved data for the bifurcaiton diagram %s", (project_dir + file_name_).c_str() );
            }
        }
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

    void execute(const std::string& file_name = {})
    {
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

        load_data(file_name);

        bool is_there_a_next_knot = knots->next();
        T_vec x_deflation; //pointer to the found deflated solution
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
                std::cin.get(); 
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
    NonlinearOperations* nonlin_op;

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
    deflate_t* deflate;
    sol_storage_def_t* sol_storage_def;
    std::string project_dir;

};


}

#endif // __DEFLATION_CONTINUATION_HPP__