#ifndef __STABILITY_CONTINUATION_HPP__
#define __STABILITY_CONTINUATION_HPP__
/**
*   The main class that checks stability of the poins, found by Deflation-Continuaiton Process (DCP). 
*   One can simply inherit from deflation_continuation, but this will trigger too much RAM used!
*/

#include <string>
#include <vector>

#include <utils/pointer_queue.h>
#include <utils/queue_fixed_size.h>

//boost serializatoin
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>

#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <deflation/solution_storage.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <deflation/solution_storage.h>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>

#include <stability/stability_analysis.hpp>

namespace main_classes
{

template<class VectorOperations, class MatrixOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver,  template<class , class , class , class > class SystemOperator>
class stability_continuation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type  T_mat;
    typedef Monitor monitor_t;
    
    typedef typename utils::queue_fixed_size<std::pair<T,T_vec>, 2> queue_t;

    typedef typename boost::archive::text_oarchive data_output;
    typedef typename boost::archive::text_iarchive data_input;


    typedef LinearSolver<LinearOperator, Preconditioner, VectorOperations, Monitor, Log> lin_slv_t;

    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;

    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        lin_slv_t
        > system_operator_t;
    
    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;
    
    typedef stability::stability_analysis<
        VectorOperations, 
        MatrixOperations,  
        NonlinearOperations, 
        LinearOperator, 
        lin_slv_t, 
        Log, 
        newton_t> stability_t;


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

    typedef utils::pointer_queue<T> queue_pointer_t;
    typedef utils::queue_fixed_size<T, 2> queue_lambda_t;
    typedef utils::queue_fixed_size<std::pair<int, int>, 2> queue_dims_t;    

public:
    stability_continuation(VectorOperations* vec_ops_, MatrixOperations* mat_ops_, VectorOperations* vec_ops_small_, MatrixOperations* mat_ops_small_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, const std::string project_dir_, unsigned int skip_files_ = 10):
            vec_ops(vec_ops_),
            mat_ops(mat_ops_),
            file_ops(file_ops_),
            log(log_),
            nonlin_op(nonlin_op_),
            project_dir(project_dir_),
            log_linsolver(log_linsolver_),
            skip_files(skip_files_)    
    {
        //set project directory the same way as in deflation_continuation
        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';

        lin_op = new LinearOperator(nonlin_op);  
        prec = new Preconditioner(nonlin_op);  

        lin_slv = new lin_slv_t(vec_ops, log_linsolver);
        lin_slv->set_preconditioner(prec);
        convergence_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, lin_slv);
        newton = new newton_t(vec_ops, system_operator, convergence_newton);
        
        stab = new stability_t(vec_ops, mat_ops, vec_ops_small_, mat_ops_small_, log, nonlin_op, lin_op, lin_slv, newton);

        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files_);

        queue_pointer = new queue_pointer_t(vec_ops->get_vector_size(), 2);
        queue_lambda = new queue_lambda_t();
        queue_dims = new queue_dims_t();

        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
        
    }
    ~stability_continuation()
    {
        delete queue_dims;
        delete queue_lambda;
        delete queue_pointer;
        delete bif_diag;
        delete stab;        
        delete newton;
        delete system_operator;
        delete convergence_newton;
        delete lin_slv;
        delete prec;
        delete lin_op;

        
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);

    }
   

    void set_linsolver(T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true)
    {
        //setup linear system:
        mon = &lin_slv->monitor();
        mon->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon->set_save_convergence_history(save_convergence_history_);
        mon->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            lin_slv->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            lin_slv->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            lin_slv->set_basis_size(basis_sz);  
//
    }

    void set_newton(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true)
    {
        convergence_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);
    }

    void set_liner_operator_stable_eigenvalues_halfplane(const T sign_)
    {
        stab->set_liner_operator_stable_eigenvalues_halfplane(sign_);
    }

    bool execute_single_curve(int& curve_number_)
    {
        typedef std::pair<bool, bool> bool2;

        int container_index = 0;

        bool2 res_read = bool2(true, true);

        while(res_read.second)
        { 
            bool curve_break = false;
            res_read = bif_diag->get_solutoin_from_curve(curve_number_, container_index, lambda_p, x_p);
            if(!res_read.first)
            {
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();                
                break;
            }
            T x_p_norm = vec_ops->norm_l2(x_p);
            if(container_index == 1)
            {
                x_p_start_norm = x_p_norm;
                lambda_p_start = lambda_p;
            }
            else
            {
                if ( (std::abs(lambda_p_start-lambda_p)<T(1.0e-6))&&(std::abs(x_p_start_norm - x_p_norm)<T(1.0e-6)) )
                {
                    curve_break = true;
                }

            }

            log->info_f("curve number = %i, index = %i, lambda = %.3lf, ||x|| = %.3le, curve_break = %d", curve_number_, container_index, lambda_p, x_p_norm, curve_break);

            if( curve_break )
            {
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
            }


            queue_pointer->push(x_p); 
            queue_lambda->push(lambda_p);
            std::pair<int, int> unstable_dim_p = stab->execute(x_p, lambda_p);
            queue_dims->push(unstable_dim_p);
            
            if( queue_pointer->is_queue_filled() )
            {
                if( queue_dims->at(0) != queue_dims->at(1) )
                {
                    stab->bisect_bifurcation_point_known(queue_pointer->at(0), queue_lambda->at(0), queue_dims->at(0), queue_pointer->at(1), queue_lambda->at(1), queue_dims->at(1), x_p, lambda_p, 21);
                    //after this (lambda_p,x_p) contain the bifurcaiton point data.
                    
                }

            }
            
        }


        return res_read.first;

    }


    void execute_all(const std::string file_name_)
    {
        load_data(file_name_);
        int curve_number = 0;
        bool get_curve = true;
        while(get_curve)
        {
            get_curve = execute_single_curve(curve_number);
            log->info_f("executing curve = %i", curve_number);
        }

    }


private:
    VectorOperations* vec_ops; 
    MatrixOperations* mat_ops;
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    std::string project_dir;
    unsigned int skip_files;
//created locally:
    LinearOperator* lin_op = nullptr;
    Preconditioner* prec = nullptr;
    lin_slv_t* lin_slv = nullptr;
    monitor_t* mon = nullptr;
    newton_t* newton = nullptr;
    convergence_newton_t* convergence_newton = nullptr;
    system_operator_t* system_operator = nullptr;
    stability_t* stab = nullptr;
    bif_diag_t* bif_diag = nullptr;
    queue_pointer_t* queue_pointer = nullptr;
    queue_lambda_t* queue_lambda = nullptr;
    queue_dims_t* queue_dims = nullptr;

//  to detect curve break during analysis
    T x_p_start_norm;
    T lambda_p_start;

    T_vec x_p;
    T lambda_p;


    bool load_data(const std::string file_name_ = {})
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

};



}

#endif // __STABILITY_CONTINUATION_HPP__