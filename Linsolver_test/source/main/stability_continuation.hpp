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
// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>

#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <deflation/solution_storage.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <deflation/solution_storage.h>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>

#include <containers/stability_diagram.h>

#include <external_libraries/lapack_wrap.h>

#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <stability/system_operator_Cayley_transform.h>
#include <stability/IRAM/iram_process.hpp>
#include <stability/stability_analysis.hpp>



namespace main_classes
{

template<class VectorOperations, class MatrixOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, class LinearOperatorShifted, class PreconditionerShifted, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator, class Parameters>
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
    // typedef typename boost::archive::xml_oarchive data_output;
    // typedef typename boost::archive::xml_iarchive data_input;

    //linear solver
    using lin_slv_t = LinearSolver<LinearOperator, Preconditioner, VectorOperations, Monitor, Log>;
    //shifted linear solver
    using lin_slv_sh_t = LinearSolver<LinearOperatorShifted, PreconditionerShifted, VectorOperations, Monitor, Log>;

    using lapack_wrap_t = lapack_wrap<T>;

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
    


    //setting up the eigensolver
    using Cayley_system_op_t = stability::system_operator_Cayley_transform<VectorOperations, NonlinearOperations, LinearOperatorShifted, lin_slv_sh_t, Log>;
    using iram_t = stability::IRAM::iram_process<VectorOperations, MatrixOperations, lapack_wrap_t, LinearOperator, Log, Cayley_system_op_t>;


    using stability_t = stability::stability_analysis<
        VectorOperations,   
        NonlinearOperations, 
        Log, 
        newton_t,
        iram_t>;


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

    typedef container::stability_diagram<
        VectorOperations, 
        VectorFileOperations, 
        Log
        > stability_diagram_t;



    typedef utils::pointer_queue<T> queue_pointer_t;
    typedef utils::queue_fixed_size<T, 2> queue_lambda_t;
    typedef utils::queue_fixed_size<std::pair<int, int>, 2> queue_dims_t;    

public:
    stability_continuation(VectorOperations* vec_ops_, MatrixOperations* mat_ops_, VectorOperations* vec_ops_small_, MatrixOperations* mat_ops_small_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, Parameters* parameters_):
            vec_ops(vec_ops_),
            mat_ops(mat_ops_),
            file_ops(file_ops_),
            log(log_),
            nonlin_op(nonlin_op_),
            log_linsolver(log_linsolver_),
            parameters(parameters_)

    {
        
        project_dir = parameters->path_to_prject;
        skip_files = parameters->deflation_continuation.skip_files;

        //set project directory the same way as in deflation_continuation
        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';

        lapack = new lapack_wrap_t(parameters->stability_continuation.Krylov_subspace);
        lin_op = new LinearOperator(nonlin_op);  
        prec = new Preconditioner(nonlin_op);  

        lin_op_sh = new LinearOperatorShifted(vec_ops, nonlin_op);
        prec_sh = new PreconditionerShifted(nonlin_op);

        lin_slv = new lin_slv_t(vec_ops, log_linsolver);
        lin_slv->set_preconditioner(prec);

        lin_slv_sh = new lin_slv_sh_t(vec_ops, log_linsolver);
        lin_slv_sh->set_preconditioner(prec_sh);

        convergence_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, lin_slv);
        newton = new newton_t(vec_ops, system_operator, convergence_newton);
        

        
        Cayley_sys_op = new Cayley_system_op_t(vec_ops, nonlin_op, lin_op_sh, lin_slv_sh, log);
        iram = new iram_t(vec_ops, mat_ops, vec_ops_small_, mat_ops_small_, lapack, lin_op, log, Cayley_sys_op);

        stab = new stability_t(vec_ops, log, nonlin_op, newton, iram);


        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files);

        stability_diagram = new stability_diagram_t(vec_ops, file_ops, log, project_dir);

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
        delete stability_diagram;
        delete bif_diag;
        delete Cayley_sys_op;
        delete iram;
        delete stab;        
        delete newton;
        delete system_operator;
        delete convergence_newton;
        delete lin_slv;
        delete lin_slv_sh;
        delete prec;
        delete lin_op;
        delete prec_sh;
        delete lin_op_sh;
        delete lapack;
        


        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);

    }
   

    void set_parameters()
    {
        set_linsolver();
        set_newton();
        set_linear_operator_stable_eigenvalues_halfplane();
    }

    void set_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true*/    
    {
        //setup linear system:
        mon = &lin_slv->monitor();
    
        T lin_solver_tol = parameters->stability_continuation.linear_solver.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->stability_continuation.linear_solver.lin_solver_max_it;
        bool save_convergence_history_ = parameters->stability_continuation.linear_solver.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->stability_continuation.linear_solver.divide_out_norms_by_rel_base;
        int use_precond_resid = parameters->stability_continuation.linear_solver.use_precond_resid;
        int resid_recalc_freq = parameters->stability_continuation.linear_solver.resid_recalc_freq;
        int basis_sz = parameters->stability_continuation.linear_solver.basis_size;

        mon->init(lin_solver_tol, T(0.0), lin_solver_max_it);
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

        //setup linear system:
        mon = &lin_slv_sh->monitor();

        mon->init(lin_solver_tol, T(0.0), lin_solver_max_it);
        mon->set_save_convergence_history(save_convergence_history_);
        mon->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            lin_slv_sh->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            lin_slv_sh->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            lin_slv_sh->set_basis_size(basis_sz);  
       
        Cayley_sys_op->set_tolerance(1.0e-9); //TODO: to json params?
        T sigma = parameters->stability_continuation.Cayley_transform_sigma_mu.at(0);
        T mu = parameters->stability_continuation.Cayley_transform_sigma_mu.at(1);
        Cayley_sys_op->set_sigma_and_mu(sigma, mu);
        iram->set_verbocity(true);
        iram->set_target_eigs("LR");
        iram->set_number_of_desired_eigenvalues(parameters->stability_continuation.desired_spectrum);
        iram->set_tolerance(1.0e-6);
        iram->set_max_iterations(100);

//
    }

    void set_newton()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        T tolerance_ = parameters->stability_continuation.newton.tolerance;
        unsigned int maximum_iterations_ = parameters->stability_continuation.newton.newton_max_it;
        T newton_wight_ = parameters->stability_continuation.newton.newton_wight;

        bool store_norms_history_ = parameters->stability_continuation.newton.store_norms_history;
        bool verbose_ = parameters->stability_continuation.newton.verbose;        

        convergence_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);
    }

    void set_linear_operator_stable_eigenvalues_halfplane()
/*const T sign_*/    
    {
        bool left_hp = parameters->stability_continuation.linear_operator_stable_eigenvalues_left_halfplane;
        
        if(left_hp)
        {
            stab->set_linear_operator_stable_eigenvalues_halfplane(T(-1.0));
            std::cout << "set_linear_operator_stable_eigenvalues_halfplane is set to -1" << std::endl;
        }
        else
        {
            stab->set_linear_operator_stable_eigenvalues_halfplane(T(1.0));
            std::cout << "set_linear_operator_stable_eigenvalues_halfplane is set to 1" << std::endl;
        }
    }


    bool execute_single_curve(int& curve_number_)
    {
        typedef std::pair<bool, bool> bool2;

        int container_index = 0;

        bool2 res_read = bool2(true, true);
        
        queue_pointer->clear();
        queue_lambda->clear();
        queue_dims->clear(); 

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
                if ( (x_p_norm == T(0.0))||(std::abs(lambda_p_start-lambda_p)<T(1.0e-6))&&(std::abs(x_p_start_norm - x_p_norm)<T(1.0e-6)) )
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
            
            try
            {
                std::pair<int, int> unstable_dim_p = stab->execute(x_p, lambda_p);
                queue_dims->push(unstable_dim_p);
                bool bifurcation_point = false;
                if( queue_pointer->is_queue_filled() )
                {
                    if( queue_dims->at(0) != queue_dims->at(1) )
                    {
                        log->info_f("container::stability_diagram: (lambda0,||x0||) = (%lf;%lf), (lambda1,||x1||) = (%lf;%lf).", double(queue_lambda->at(0)), vec_ops->norm(queue_pointer->at(0)), double(queue_lambda->at(1)), vec_ops->norm(queue_pointer->at(1)) );
                        stab->bisect_bifurcation_point_known(queue_pointer->at(0), queue_lambda->at(0), queue_dims->at(0), queue_pointer->at(1), queue_lambda->at(1), queue_dims->at(1), x_p, lambda_p, 20);
                        //after this (lambda_p,x_p) contain the bifurcaiton point data.
                        bifurcation_point = true;
                    }

                }

                if(bifurcation_point)
                    stability_diagram->add(lambda_p, unstable_dim_p.first, unstable_dim_p.second, x_p);
                else
                    stability_diagram->add(lambda_p, unstable_dim_p.first, unstable_dim_p.second);

            }
            catch(const std::exception& e)
            {
                log->warning_f("container::stability_diagram: %s. Point was not added.", e.what());
                log->warning_f("container::stability_diagram: failed for lambda_p = %lf, queue is cleared.", lambda_p);
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
            }

            
        }


        return res_read.first;

    }


    void edit()
    {
        std::string file_name_stability_ = parameters->stability_diagram_file_name;
        
        bool stability_data = load_stability_data(file_name_stability_);  
        if(stability_data)
        {
            std::cout << "entering interactive edit mode" << std::endl;
            std::cout << "enter 'd' to pop_back() the curve or 'q' to quit." << std::endl;
            char c = 'c';
            while(c != 'q')
            {
                std::cout << "file " << file_name_stability_ << " contains:" << std::endl;
                stability_diagram->print_curves_status();
                c = std::cin.get();
                if(c=='d')
                {
                    stability_diagram->pop_back_curve();
                }
            }
            c = std::cin.get();
            std::cout << "save file(y/n)>>>";
            c = std::cin.get();
            if(c == 'y')
                save_stability_data(file_name_stability_);

        }
        else
        {
            log->warning_f("MAIN:stability_diagram: file %s doesn't exist; called edit with no file provided!", file_name_stability_.c_str());
        }

    }


    void execute()
    {
        std::string file_name_diagram_ = parameters->bifurcaiton_diagram_file_name;
        std::string file_name_stability_ = parameters->stability_diagram_file_name;

        bool file_exists = load_diagram_data(file_name_diagram_);
        if(file_exists)
        {
            bool stability_data = load_stability_data(file_name_stability_);
            int curve_number = 0;
            if(stability_data)
            {
                curve_number = stability_diagram->current_curve();
            }
            bool get_curve = true;
            while(get_curve)
            {
                log->info_f("executing curve = %i", curve_number);
                stability_diagram->open_curve(curve_number);
                get_curve = execute_single_curve(curve_number);
                if(get_curve)
                {
                    stability_diagram->close_curve();
                    save_stability_data(file_name_stability_);
                }
                

            }
        }

    }


private:
    VectorOperations* vec_ops; 
    MatrixOperations* mat_ops;
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    Parameters* parameters;
    std::string project_dir;
    unsigned int skip_files;
//created locally:
    lapack_wrap_t* lapack = nullptr;
    LinearOperator* lin_op = nullptr;
    LinearOperatorShifted* lin_op_sh = nullptr;
    Preconditioner* prec = nullptr;
    PreconditionerShifted* prec_sh = nullptr;
    lin_slv_t* lin_slv = nullptr;
    lin_slv_sh_t* lin_slv_sh = nullptr;

    Cayley_system_op_t* Cayley_sys_op = nullptr;
    iram_t* iram = nullptr;
    monitor_t* mon = nullptr;
    newton_t* newton = nullptr;
    convergence_newton_t* convergence_newton = nullptr;
    system_operator_t* system_operator = nullptr;
    stability_t* stab = nullptr;
    bif_diag_t* bif_diag = nullptr;
    queue_pointer_t* queue_pointer = nullptr;
    queue_lambda_t* queue_lambda = nullptr;
    queue_dims_t* queue_dims = nullptr;
    stability_diagram_t* stability_diagram = nullptr;

//  to detect curve break during analysis
    T x_p_start_norm;
    T lambda_p_start;

    T_vec x_p;
    T lambda_p;


    bool load_diagram_data(const std::string file_name_ = {})
    {
        bool file_exists = false;
        if(!file_name_.empty())
        {
            std::ifstream load_file( (project_dir + file_name_).c_str() );
            if(load_file.good())
            {
                log->info_f("MAIN:stability_continuation: reading data for the bifurcaiton diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                ia >> (*bif_diag);
                load_file.close();
                log->info_f("MAIN:stability_continuation: read data for the bifurcaiton diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:stability_continuation: failed to load saved data for the bifurcaiton diagram %s", (project_dir + file_name_).c_str() );
                file_exists = false;
            }
        }
        return file_exists;
    }

    bool load_stability_data(const std::string file_name_ = {})
    {
        bool file_exists = false;
        if(!file_name_.empty())
        {
            std::ifstream load_file( (project_dir + file_name_).c_str() );  
            if(load_file.good())
            {
                log->info_f("MAIN:stability_continuation: reading data for the stability diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                ia >> (*stability_diagram);
                load_file.close();
                log->info_f("MAIN:stability_continuation: read data for the stability diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:stability_continuation: failed to load stability saved data for the stability diagram %s", (project_dir + file_name_).c_str() );
                file_exists = false;                
            }

        }        
        return file_exists;
    }


    void save_stability_data(const std::string& file_name_ = {})
    {
        if(!file_name_.empty())
        {
            log->info_f("MAIN:stability_continuation: saving data for the stability diagram in %s ...", (project_dir + file_name_).c_str() );
            std::ofstream save_file( (project_dir + file_name_).c_str() );
            data_output oa(save_file);
            oa << (*stability_diagram);
            save_file.close();
            log->info_f("MAIN:stability_continuation: saved data for the stability diagram in %s", (project_dir + file_name_).c_str() );
        }        
    }

};



}

#endif // __STABILITY_CONTINUATION_HPP__