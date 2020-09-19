#ifndef __PLOT_DIAGRAM_TO_POS_HPP__
#define __PLOT_DIAGRAM_TO_POS_HPP__
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

#include <containers/stability_diagram.h>
#include <common/pos_bif_diag_output.h>


// for sleep?
#include <thread>
#include <chrono>

namespace main_classes
{




template<class VectorOperations, class MatrixOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver,  template<class , class , class , class > class SystemOperator, class Parameters>
class plot_diagram_to_pos
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


    //types of points in different curves
    typedef typename bif_diag_t::curve_point_type solution_point_t;
    typedef typename stability_diagram_t::stability_point_type stability_point_t;


    struct solution_and_stability_point:public solution_point_t
    {
        int dim_unstable;

    };


public:
    plot_diagram_to_pos(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, Parameters* parameters_):
            vec_ops(vec_ops_),
            file_ops(file_ops_),
            log(log_),
            nonlin_op(nonlin_op_),
            log_linsolver(log_linsolver_),
            parameters(parameters_)
    {
        project_dir = parameters->path_to_prject;
        //set project directory the same way as in deflation_continuation
        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';

        save_diagram_dir = project_dir + std::string("diagram/");

        lin_op = new LinearOperator(nonlin_op);  
        prec = new Preconditioner(nonlin_op);  

        lin_slv = new lin_slv_t(vec_ops, log_linsolver);
        lin_slv->set_preconditioner(prec);
        convergence_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, lin_slv);
        newton = new newton_t(vec_ops, system_operator, convergence_newton);
        
        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir);

        stability_diagram = new stability_diagram_t(vec_ops, file_ops, log, project_dir);

        queue_pointer = new queue_pointer_t(vec_ops->get_vector_size(), 2);
        queue_lambda = new queue_lambda_t();
        queue_dims = new queue_dims_t();


        vec_ops->init_vector(x_p); vec_ops->start_use_vector(x_p);
        vec_ops->init_vector(b_pos_plot); vec_ops->start_use_vector(b_pos_plot);

    }
    ~plot_diagram_to_pos()
    {
        delete queue_dims;
        delete queue_lambda;
        delete queue_pointer;
        delete stability_diagram;
        delete bif_diag;        
        delete newton;
        delete system_operator;
        delete convergence_newton;
        delete lin_slv;
        delete prec;
        delete lin_op;

        
        vec_ops->stop_use_vector(x_p); vec_ops->free_vector(x_p);
        vec_ops->stop_use_vector(b_pos_plot); vec_ops->free_vector(b_pos_plot);

    }
   

    void set_parameters()
    {
        set_linsolver();
        set_newton();
        set_plot_pos_sols( parameters->plot_solutions.plot_solution_frequency );
    }


    void set_linsolver()
/*
T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true
*/    
    {
        //setup linear system:
        mon = &lin_slv->monitor();

        T lin_solver_tol = parameters->nonlinear_operator.linear_solver.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->nonlinear_operator.linear_solver.lin_solver_max_it;
        bool save_convergence_history_ = parameters->nonlinear_operator.linear_solver.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->nonlinear_operator.linear_solver.divide_out_norms_by_rel_base;
        int use_precond_resid = parameters->nonlinear_operator.linear_solver.use_precond_resid;
        int resid_recalc_freq = parameters->nonlinear_operator.linear_solver.resid_recalc_freq;
        int basis_sz = parameters->nonlinear_operator.linear_solver.basis_size;

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

    void set_newton()
/*
T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true
*/    
    {
        T tolerance_ = parameters->nonlinear_operator.newton.tolerance;
        unsigned int maximum_iterations_ = parameters->nonlinear_operator.newton.newton_max_it;
        T newton_wight_ = parameters->nonlinear_operator.newton.newton_wight;

        bool store_norms_history_ = parameters->nonlinear_operator.newton.store_norms_history;
        bool verbose_ = parameters->nonlinear_operator.newton.verbose;

        convergence_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);
    }


    bool execute_single_curve(int& curve_number_, bool use_stability_ = false)
    {
        
        std::vector<solution_point_t> curve = bif_diag->get_curve_points_vector(curve_number_);
        if(curve.size() == 0)
        {
            return false;
        }
        else
        {
            if(!use_stability_)
            {
                std::string file_name;
                
                int total_norms = curve.at(0).vector_norms.size();
                for(int j=0;j<total_norms-1;j++)
                {
                    file_name = std::to_string(j) + "_" + std::to_string(j+1) + "_curve_" + std::to_string(curve_number_) + ".pos";
                    
                    file_operations::plot_diagram_no_stability<T, solution_point_t>(save_diagram_dir, file_name, curve_number_, curve, j, j+1);
                }         

            }
            else
            {
                std::vector<solution_and_stability_point> curve_stability;
                curve_stability.reserve(curve.size());

                for(auto &x: curve)
                {
                    solution_and_stability_point aaa;
                    static_cast<solution_point_t&>(aaa) = x;
                    curve_stability.push_back(aaa);
                }


                std::vector<stability_point_t> stability_curve = stability_diagram->get_curve_points_vector(curve_number_);


                std::deque<stability_point_t> qqq( std::deque<stability_point_t>(stability_curve.begin(), stability_curve.end() ) );

                std::vector<solution_and_stability_point> bifurcation_points;

                T lambda_pr = 0;
                stability_point_t stab_point_front = qqq.front();
                for(auto &c: curve_stability)
                {
                    stability_point_t stab_point_front_attempt = qqq.front();
                    T lambda_s = stab_point_front_attempt.lambda;
                    if( c.is_data_avaliable && (c.lambda == stab_point_front_attempt.lambda) )
                    {
                        stab_point_front = qqq.front();
                        qqq.pop_front();
                        
                    }    
                    else if( (stab_point_front_attempt.point_type == "bifurcation")&&(( lambda_pr-lambda_s)*(c.lambda - lambda_s)<T(0.0)) )
                    {
                        if(qqq.size()>1)
                        {

                            bifurcation_points.push_back(form_bifurcation_point(curve_number_, stab_point_front_attempt));

                            stab_point_front = qqq.front(); 
                            qqq.pop_front();
                            
                        }
                        else
                        {
                            bifurcation_points.push_back(form_bifurcation_point(curve_number_, stab_point_front_attempt));
                            stab_point_front = qqq.front();
                            
                        }
                   
                    } 
                    lambda_pr = c.lambda;
                    int dim_unstable = stab_point_front.unstable_dim_R + 2*stab_point_front.unstable_dim_C;
                    c.dim_unstable = dim_unstable - 1;



                }

                std::string file_name;
                
                int total_norms = curve_stability.at(0).vector_norms.size();
                for(int j=0;j<total_norms-1;j++)
                {
                    file_name = std::to_string(j) + "_" + std::to_string(j+1) + "_curve_" + std::to_string(curve_number_) + ".pos";
                    
                    file_operations::plot_diagram_stability<T, solution_and_stability_point>(save_diagram_dir, file_name, curve_number_, curve_stability, bifurcation_points, j, j+1);
                
                }

                file_name = "_curve_" + std::to_string(curve_number_) + ".dat";
                file_operations::plot_diagram_stability_gnuplot<T, solution_and_stability_point>(save_diagram_dir, file_name, curve_number_, curve_stability, bifurcation_points);       


            }
            
            if(plot_pos_sols>0)
            {
                FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
                std::string gnuplot_command = "set term png size 1900,1024";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                gnuplot_command = "unset colorbox";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                gnuplot_command = "unset border";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                gnuplot_command = "unset xtics";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                gnuplot_command = "unset ytics";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                gnuplot_command = "unset key";
                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());                
                for( auto &x: curve)
                {
                    if(x.is_data_avaliable)
                    {
                        auto file_name = x.id_file_name;
                        if( (file_name-1)%plot_pos_sols == 0)
                        {
                            std::string input_file_name = project_dir+std::to_string(curve_number_) + "/" + std::to_string(file_name);
                            
                            //std::string output_file_name = save_diagram_dir + "solutions/" + "curve_" + std::to_string(curve_number_) + "solution_" + std::to_string(file_name) + ".pos";

                            std::string output_file_name_data = save_diagram_dir + "solutions/" + "data.dat";

                            std::string output_file_name_image = save_diagram_dir + "solutions/" + "curve_" + std::to_string(curve_number_) + "_solution_" + std::to_string(file_name) + ".png";

                            try
                            {
                                file_ops->read_vector(input_file_name, b_pos_plot);
                                
                                //nonlin_op->write_solution(output_file_name, (T_vec&)b_pos_plot);
                                //write_solution_plane dumps the solution as it is.
                                nonlin_op->write_solution_plain(output_file_name_data, (T_vec&)b_pos_plot);
                                
                                gnuplot_command = "set output '" + output_file_name_image + "'";
                                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                                gnuplot_command = "plot '" + output_file_name_data + "' matrix with image";
                                fprintf(gnuplotPipe, "%s \n", gnuplot_command.c_str());
                                fflush(gnuplotPipe);
                                // this is very bad!!!
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                            }
                            catch(...)
                            {
                                log->warning_f("MAIN:plot_diagram_to_pos: cannot read data file %s while saving physical solution. Skipping it", input_file_name.c_str() );
                            }

                        }
                    }
                }
                pclose(gnuplotPipe);
            }

            curve_number_++;
            return true;
        }
    }


    void set_plot_pos_sols(int plot_pos_sols_)
    {
        plot_pos_sols = plot_pos_sols_;
    }

    void execute()
    {
        
        std::string file_name_diagram_ = parameters->bifurcaiton_diagram_file_name;
        std::string file_name_stability_ = parameters->stability_diagram_file_name;

        bool file_exists = load_diagram_data(file_name_diagram_);
        if(file_exists)
        {
            bool stability_data = load_stability_data(file_name_stability_);

            bool get_curve = true;
            int curve_number = 0;

            while(get_curve)
            {
                log->info_f("executing curve = %i", curve_number);
                get_curve = execute_single_curve(curve_number, stability_data);

            }
            log->info("done plotting curves");
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
    std::string save_diagram_dir;
    Parameters* parameters;
    unsigned int skip_files;
//created locally:
    LinearOperator* lin_op = nullptr;
    Preconditioner* prec = nullptr;
    lin_slv_t* lin_slv = nullptr;
    monitor_t* mon = nullptr;
    newton_t* newton = nullptr;
    convergence_newton_t* convergence_newton = nullptr;
    system_operator_t* system_operator = nullptr;
    bif_diag_t* bif_diag = nullptr;
    queue_pointer_t* queue_pointer = nullptr;
    queue_lambda_t* queue_lambda = nullptr;
    queue_dims_t* queue_dims = nullptr;
    stability_diagram_t* stability_diagram = nullptr;

//  to detect curve break during analysis
    T_vec x_p;
    T_vec b_pos_plot;
    int plot_pos_sols = -1;


    bool load_diagram_data(const std::string file_name_ = {})
    {
        bool file_exists = false;
        if(!file_name_.empty())
        {
            std::ifstream load_file( (project_dir + file_name_).c_str() );
            if(load_file.good())
            {
                log->info_f("MAIN:plot_diagram_to_pos: reading data for the bifurcaiton diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                ia >> (*bif_diag);
                load_file.close();
                log->info_f("MAIN:plot_diagram_to_pos: read data for the bifurcaiton diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:plot_diagram_to_pos: failed to load saved data for the bifurcaiton diagram %s", (project_dir + file_name_).c_str() );
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
                log->info_f("MAIN:plot_diagram_to_pos: reading data for the stability diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                ia >> (*stability_diagram);
                load_file.close();
                log->info_f("MAIN:plot_diagram_to_pos: read data for the stability diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:plot_diagram_to_pos: failed to load stability saved data for the stability diagram %s", (project_dir + file_name_).c_str() );
                file_exists = false;                
            }

        }        
        return file_exists;
    }


    solution_and_stability_point form_bifurcation_point(const int curve_number_, const stability_point_t& stab_point_front)
    {
        stability_diagram->get_solution_from_record(curve_number_, stab_point_front, x_p);
        std::vector<T> bif_diag_norms;
        solution_and_stability_point bif_point;
        nonlin_op->norm_bifurcation_diagram(x_p, bif_diag_norms);
        bif_point.lambda = stab_point_front.lambda;
        bif_point.is_data_avaliable = true;
        bif_point.vector_norms = bif_diag_norms;
        bif_point.id_file_name = stab_point_front.id_file_name;
        bif_point.dim_unstable = stab_point_front.unstable_dim_R + stab_point_front.unstable_dim_C - 1;
        return(bif_point);
    }


};



}

#endif // __PLOT_DIAGRAM_TO_POS_HPP__