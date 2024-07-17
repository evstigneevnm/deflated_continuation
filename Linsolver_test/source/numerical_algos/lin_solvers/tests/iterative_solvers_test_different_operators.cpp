#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <common/cpu_vector_operations.h>
#include "linear_operator_advection.h"
#include "linear_operator_diffusion.h"
#include "linear_operator_elliptic.h"
#include "preconditioner_advection.h"
#include "preconditioner_diffusion.h"
#include "preconditioner_elliptic.h"
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>
// #include "residual_regularization_test.h"

#define M_PIl 3.141592653589793238462643383279502884L



int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = SCALAR_TYPE;
    using vec_ops_t = cpu_vector_operations<T>;
    using T_vec = typename vec_ops_t::vector_type;
    using lin_op_adv_t = tests::linear_operator_advection<vec_ops_t, log_t>;
    using lin_op_diff_t = tests::linear_operator_diffusion<vec_ops_t, log_t>;
    using lin_op_elliptic_t = tests::linear_operator_elliptic<vec_ops_t, log_t>;
    using prec_adv_t = tests::preconditioner_advection<vec_ops_t, lin_op_adv_t, log_t>;
    using prec_diff_t = tests::preconditioner_diffusion<vec_ops_t, lin_op_diff_t, log_t>;
    using prec_elliptic_t = tests::preconditioner_elliptic<vec_ops_t, lin_op_elliptic_t, log_t>;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;
    using gmres_adv_t = numerical_algos::lin_solvers::gmres<lin_op_adv_t, prec_adv_t, vec_ops_t, monitor_t, log_t>;
    using gmres_diff_t = numerical_algos::lin_solvers::gmres<lin_op_diff_t, prec_diff_t, vec_ops_t, monitor_t, log_t>;
    // using gmres_adv_noprec_t = numerical_algos::lin_solvers::gmres<lin_op_adv_t, vec_ops_t, monitor_t, log_t >;
    // using gmres_diff_noprec_t = numerical_algos::lin_solvers::gmres< lin_op_diff_t, vec_ops_t, monitor_t, log_t >;
    // using residual_reg_t = numerical_algos::lin_solvers::detail::residual_regularization_test<vec_ops_t, log_t>;
    // using gmres_elliptic_w_reg_t = numerical_algos::lin_solvers::gmres< vec_ops_t, monitor_t, log_t, lin_op_elliptic_t, prec_elliptic_t, residual_reg_t>;



    int error = 0;
    log_t log;
    log.info("test gmres");
    std::size_t N_with_preconds = 500;
    std::size_t N_with_no_preconds = 50;
    std::shared_ptr<vec_ops_t> vec_ops;

    auto get_residual = [&log, &vec_ops](auto& A, auto& x, auto &y) 
    {
        T_vec resid;
        vec_ops->init_vector(resid);
        vec_ops->start_use_vector(resid);
        A.apply(x,resid);
        vec_ops->add_lin_comb(1,y,-1,resid);
        log.info_f("||Lx-y|| = %e", vec_ops->norm(resid) );
        vec_ops->stop_use_vector(resid);
        vec_ops->free_vector(resid);
    };

    //testing left and right preconditioners
    {
        std::size_t N = N_with_preconds;
        vec_ops = std::make_shared<vec_ops_t>(N);
        prec_diff_t prec_diff(vec_ops, 15);
        prec_adv_t prec_adv(vec_ops, 1);

        T tau = 1.0;
        T a = 1.0;
        T_vec x,y,resid;
        vec_ops->init_vector(x);
        vec_ops->init_vector(y);        
        vec_ops->start_use_vector(x);
        vec_ops->start_use_vector(y);

        log.info_f("=>diffusion with size %i, timestep %.02f.", vec_ops->size(), tau ); 
        lin_op_diff_t lin_op_diff(*vec_ops, tau); //with time step 5


        for(int j=0;j<N;j++)
        {
            y[j] = std::sin(1.0*j/(N-1)*M_PIl);
            // x[j] = 0.1*std::sin(1.0*j/(N-1)*M_PIl);
        }
        y[0] = y[N-1] = 0;

        gmres_diff_t::params params_diff;
        T rel_tol = 1.0e-10;
        int max_iters_num = 300;
        // params_diff.monitor.rel_tol = 1.0e-10;
        // params_diff.monitor.max_iters_num = 300;
        params_diff.basis_size = 15;
        {
            log.info("left preconditioner");
            params_diff.preconditioner_side = 'L';
            gmres_diff_t gmres(vec_ops.get(), &log, params_diff);
            gmres.set_preconditioner(&prec_diff);
            auto mon = &gmres.monitor();
            mon->init(rel_tol, 0, max_iters_num);
            mon->set_save_convergence_history(true);
            mon->set_divide_out_norms_by_rel_base(true);    

            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(lin_op_diff, y, x);
            error += (!res);

            log.info_f("pLgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(lin_op_diff, y, x);
            error += (!res);
            log.info_f("pLgmres res with x0: %s", res?"true":"false");
            get_residual(lin_op_diff, x, y);
        }
        {
            log.info("right preconditioner");
            params_diff.preconditioner_side = 'R';
            gmres_diff_t gmres(vec_ops.get(), &log, params_diff);
            gmres.set_preconditioner(&prec_diff);
            auto mon = &gmres.monitor();
            mon->init(rel_tol, 0, max_iters_num);
            mon->set_save_convergence_history(true);
            mon->set_divide_out_norms_by_rel_base(true);              
            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(lin_op_diff, y, x);
            error += (!res);
            log.info_f("pRgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(lin_op_diff, y, x);
            error += (!res);
            log.info_f("pRgmres res with x0: %s", res?"true":"false");
            get_residual(lin_op_diff, x, y);
        }
        
        log.info_f("=>advection with size %i, speed %.02f, timestep %.02f.", vec_ops->size(), a, tau ); 
        gmres_adv_t::params params_adv;
        rel_tol = 1.0e-10;
        max_iters_num = 300;
        params_adv.basis_size = 15;    
        lin_op_adv_t lin_op_adv(*vec_ops, a, tau);
        {
            gmres_adv_t gmres(vec_ops.get(), &log, params_adv);
            gmres.set_preconditioner(&prec_adv);
            auto mon = &gmres.monitor();
            mon->init(rel_tol, 0, max_iters_num);
            mon->set_save_convergence_history(true);
            mon->set_divide_out_norms_by_rel_base(true);              

            log.info("left preconditioner");
            params_adv.preconditioner_side = 'L';
            vec_ops->assign_scalar(0.0, x);       
            bool res = gmres.solve(lin_op_adv, y, x);
            error += (!res);
            log.info_f("pLgmres res: %s", res?"true":"false");
            log.info("reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.9999999, x);
            res = gmres.solve(lin_op_adv, y, x);
            error += (!res);
            log.info_f("pLgmres res with x0: %s", res?"true":"false");
            get_residual(lin_op_adv, x, y);             
        }
        {
            gmres_adv_t gmres(vec_ops.get(), &log, params_adv);
            gmres.set_preconditioner(&prec_adv);
            auto mon = &gmres.monitor();
            mon->init(rel_tol, 0, max_iters_num);
            mon->set_save_convergence_history(true);
            mon->set_divide_out_norms_by_rel_base(true);              
            log.info("right preconditioner");
            params_adv.preconditioner_side = 'R';
            vec_ops->assign_scalar(0.0, x);       
            bool res = gmres.solve(lin_op_adv, y, x);
            error += (!res);
            log.info_f("pRgmres res: %s", res?"true":"false");
            log.info("reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.9999999, x);
            res = gmres.solve(lin_op_adv, y, x);
            error += (!res);
            log.info_f("pRgmres res with x0: %s", res?"true":"false");
            get_residual(lin_op_adv, x, y);             
            // for(int j=0;j<N;j++)
            // {
            //     std::cout << (1.0*j+0.5)/N << "," << x[j] << std::endl;
            // }

        }

        vec_ops->stop_use_vector(x);
        vec_ops->stop_use_vector(y);
        vec_ops->free_vector(x);
        vec_ops->free_vector(y);
    }
    // {
    //     std::size_t N = N_with_no_preconds;
    //     vec_ops = std::make_shared<vec_ops_t>(N);
    //     T tau = 1.0;
    //     T a = 1.0;
    //     T_vec x,y,resid;
    //     vec_ops->init_vector(x);
    //     vec_ops->init_vector(y);        
    //     vec_ops->start_use_vector(x);
    //     vec_ops->start_use_vector(y);

    //     log.info_f("=>diffusion with size %i, timestep %.02f.", vec_ops->size(), tau ); 
    //     auto lin_op_diff = std::make_shared<lin_op_diff_t>(*vec_ops, tau); //with time step 5

    //     for(int j=0;j<N;j++)
    //     {
    //         y[j] = std::sin(1.0*j/(N-1)*M_PIl);
    //         // x[j] = 0.1*std::sin(1.0*j/(N-1)*M_PIl);
    //     }
    //     y[0] = y[N-1] = 0;
    //     gmres_diff_noprec_t::params params_diff;
    //     params_diff.monitor.rel_tol = 1.0e-10;
    //     params_diff.monitor.max_iters_num = 50;
    //     params_diff.basis_size = 30;        
    //     gmres_diff_noprec_t gmres_diff(lin_op_diff, vec_ops, &log, params_diff);

    //     vec_ops->assign_scalar(0.0, x);
    //     log.info("no preconditioner");        
    //     bool res = gmres_diff.solve(y, x);
    //     error += (!res);

    //     log.info_f("gmres res: %s", res?"true":"false");
    //     log.info("reusing the solution...");
    //     vec_ops->add_mul_scalar(0.0, 0.9999999, x);
    //     res = gmres_diff.solve(y, x);
    //     error += (!res);
    //     log.info_f("gmres res with x0: %s", res?"true":"false");
    //     get_residual(*lin_op_diff, x, y);        


    //     log.info_f("=>advection with size %i, speed %.02f, timestep %.02f.", vec_ops->size(), a, tau ); 
    //     gmres_adv_noprec_t::params params_adv;
    //     params_adv.monitor.rel_tol = 1.0e-10;
    //     params_adv.monitor.max_iters_num = 50;
    //     params_adv.basis_size = 50;    

    //     auto lin_op_adv = std::make_shared<lin_op_adv_t>(*vec_ops, a, tau);
    //     gmres_adv_noprec_t gmres_adv(lin_op_adv, vec_ops, &log, params_adv);
    //     vec_ops->assign_scalar(0.0, x);
    //     log.info("no preconditioner");        
    //     res = gmres_adv.solve(y, x);
    //     error += (!res);
    //     log.info_f("gmres res: %s", res?"true":"false");
    //     log.info("reusing the solution...");
    //     vec_ops->add_mul_scalar(0.0, 0.9999999, x);
    //     res = gmres_adv.solve(y, x);
    //     error += (!res);
    //     log.info_f("gmres res with x0: %s", res?"true":"false");
    //     get_residual(*lin_op_adv, x, y);  
    //     // for(int j=0;j<N;j++)
    //     // {
    //     //     std::cout << (1.0*j+0.5)/N << "," << x[j] << std::endl;
    //     // }
    //     vec_ops->stop_use_vector(x);
    //     vec_ops->stop_use_vector(y);
    //     vec_ops->free_vector(x);
    //     vec_ops->free_vector(y);

    // }
    



    //testing elliptic operator with constant kernel
    // {
    //     std::size_t N = N_with_preconds;
    //     vec_ops = std::make_shared<vec_ops_t>(N);
    //     auto prec_elliptic = std::make_shared<prec_elliptic_t>(vec_ops, 15);
    //     auto residual_reg = std::make_shared<residual_reg_t>(vec_ops);//, &log); //use log to see the action of the residual regularization

    //     T_vec x,y,resid;
    //     vec_ops->init_vector(x);
    //     vec_ops->init_vector(y);        
    //     vec_ops->start_use_vector(x);
    //     vec_ops->start_use_vector(y);

    //     log.info_f("=>elliptic with size %i", vec_ops->size() ); 
    //     auto lin_op_elliptic = std::make_shared<lin_op_elliptic_t>(*vec_ops); //with time step 5


    //     for(int j=0;j<N;j++)
    //     {
    //         y[j] = std::sin(2.0*j/(N-1)*M_PIl);
    //     }

    //     gmres_elliptic_w_reg_t::params params_elliptic;
    //     params_elliptic.monitor.rel_tol = 1.0e-10;
    //     params_elliptic.monitor.max_iters_num = 300;
    //     params_elliptic.basis_size = 25;
    //     {
    //         log.info("left preconditioner");
    //         params_elliptic.preconditioner_side = 'L';
    //         gmres_elliptic_w_reg_t gmres(lin_op_elliptic, vec_ops, &log, params_elliptic, prec_elliptic, residual_reg);

    //         vec_ops->assign_scalar(0.0, x);
    //         bool res = gmres.solve(y, x);
    //         error += (!res);

    //         log.info_f("pLgmres res: %s", res?"true":"false");
    //         log.info(" reusing the solution...");
    //         vec_ops->add_mul_scalar(0.0, 0.99999, x);
    //         res = gmres.solve(y, x);
    //         error += (!res);
    //         log.info_f("pLgmres res with x0: %s", res?"true":"false");
    //         log.info_f("solution final norm = %e", vec_ops->norm(x) );
    //         get_residual(*lin_op_elliptic, x, y);
    //     }
    //     {
    //         log.info("right preconditioner");
    //         params_elliptic.preconditioner_side = 'R';
    //         gmres_elliptic_w_reg_t gmres(lin_op_elliptic, vec_ops, &log, params_elliptic, prec_elliptic, residual_reg);

    //         vec_ops->assign_scalar(0.0, x);
    //         bool res = gmres.solve(y, x);
    //         error += (!res);
    //         log.info_f("pRgmres res: %s", res?"true":"false");
    //         log.info(" reusing the solution...");
    //         vec_ops->add_mul_scalar(0.0, 0.99999, x);
    //         res = gmres.solve(y, x);
    //         error += (!res);
    //         log.info_f("pRgmres res with x0: %s", res?"true":"false");
    //         log.info_f("solution final norm = %e", vec_ops->norm(x) );
    //         get_residual(*lin_op_elliptic, x, y);
    //     }
        
    //     vec_ops->stop_use_vector(x);
    //     vec_ops->stop_use_vector(y);
    //     vec_ops->free_vector(x);
    //     vec_ops->free_vector(y);
    // }

 
    if(error > 0)
    {
        log.error_f("Got error = %e.", error ) ;
    }
    else
    {
        log.info("No errors.") ;   
    }

    return error;
}