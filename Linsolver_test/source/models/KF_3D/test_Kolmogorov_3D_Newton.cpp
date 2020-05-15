#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>

//#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator_time_globalization.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>



#define Blocks_x_ 32
#define Blocks_y_ 16



int main(int argc, char const *argv[])
{
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> KF_3D_t;
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    if(argc != 4)
    {
        std::cout << argv[0] << " alpha R N:\n 0<alpha<=1, R is the Reynolds number, N = 2^n- discretization in one direction\n";
        return(0);       
    }
    
    real alpha = std::atof(argv[1]);
    real Rey = std::atof(argv[2]);
    size_t N = std::atoi(argv[3]);
    int one_over_alpha = int(1/alpha);

    init_cuda(-1);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Using alpha = " << alpha << ", Reynolds = " << Rey << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    //linsolver control
    unsigned int lin_solver_max_it = 1000;
    real lin_solver_tol = 5.0e-3;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;
    //newton deflation control
    unsigned int newton_def_max_it = 250;
    real newton_def_tol = 1.0e-10;



    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(3*(Nx*Ny*Mz-1), CUBLAS);
    
    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);
    // linear solver config
    typedef utils::log_std log_t;
    typedef numerical_algos::lin_solvers::default_monitor<
        gpu_vector_operations_t,log_t> monitor_t;
    typedef nonlinear_operators::linear_operator_K_3D<
        gpu_vector_operations_t, KF_3D_t> lin_op_t;
    typedef nonlinear_operators::preconditioner_K_3D<
        gpu_vector_operations_t, KF_3D_t, lin_op_t> prec_t;    
    typedef numerical_algos::lin_solvers::bicgstabl<
        lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t> lin_solver_t;

    monitor_t *mon;

    log_t log;
    log_t log3;
    log3.set_verbosity(1);
    lin_op_t lin_op(KF_3D);
    prec_t prec(KF_3D);    

    lin_solver_t lin_solver(vec_ops, &log3);
    lin_solver.set_preconditioner(&prec);

    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver.set_basis_size(basis_sz);
    mon = &lin_solver.monitor();
    mon->init(lin_solver_tol, real(0.f), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);   


    vec x0, x1, dx, x_back, b;

    vec_ops->init_vector(b); vec_ops->start_use_vector(b);
    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
    vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
    vec_ops->init_vector(x_back); vec_ops->start_use_vector(x_back);
    

    
    KF_3D->randomize_vector(x0);
    vec_ops->assign(x0, x_back);
    printf("initial solution norm = %le, div = %le\n", vec_ops->norm(x0), KF_3D->div_norm(x0));

    real solution_norm = 1;
    unsigned int iter = 0;
    real mu_min = 0.005;
    real mu_0 = 1.0;
    real mu = mu_0;
    bool ok_flag = true;
    std::vector<real> newton_norm;

    KF_3D->F(x0, Rey, b);
    solution_norm = vec_ops->norm(b);
    newton_norm.push_back(solution_norm);
    
    FILE *stream;
    stream=fopen("newton_convergence.dat", "w" );
    fprintf(stream, "%le\n", solution_norm );
    while((solution_norm > newton_def_tol)&&(iter < newton_def_max_it)&&(ok_flag))
    {
        vec_ops->assign_scalar(0.0, dx);
        KF_3D->F(x0, Rey, b);
        real solution_norm0 = vec_ops->norm_l2(b);
        vec_ops->add_mul_scalar(0.0, -1.0, b); // b:=-F(x0)
        KF_3D->set_linearization_point(x0, Rey);
        bool res_flag_ = lin_solver.solve(lin_op, b, dx);

        real norm_ratio = 10;
        mu = mu_0;
        
        while(norm_ratio>2.0)
        {
            vec_ops->assign_mul(1.0, x0, mu, dx, x1);
            KF_3D->F(x1, Rey, b);
            solution_norm = vec_ops->norm_l2(b);
            norm_ratio = solution_norm/solution_norm0;
            printf("mu = %le\n", mu);
            if(mu<mu_0)
                std::cin.get();             

            mu*=0.5;
            if(mu<mu_min)
            {
                ok_flag = false;
                break;
            }


        }
        vec_ops->assign(x1,x0);

        iter++;

        printf("linearization solution norm = %le, div = %le\n", vec_ops->norm(x0), KF_3D->div_norm(x0));
        printf("update norm = %le, div = %le\n", vec_ops->norm(dx), KF_3D->div_norm(dx));
        printf("RHS solution norm = %le->%le, div = %le\n", solution_norm0, solution_norm , KF_3D->div_norm(b));


        newton_norm.push_back(solution_norm);
        fprintf(stream, "%le\n", solution_norm );
        fflush(stream);
    }
    if(!ok_flag)
    {
        printf("Failed to converge!\n");
    }
    fclose(stream);

    printf("returned solution norm = %le, div = %le\n", vec_ops->norm(x0), KF_3D->div_norm(x0));
    KF_3D->write_solution_abs("x_1.pos", x0);


    // testing newton with convergence strategy
    typedef nonlinear_operators::newton_method::convergence_strategy<
        gpu_vector_operations_t, 
        KF_3D_t, 
        log_t> convergence_newton_t;
    
    // typedef nonlinear_operators::system_operator<
    //     gpu_vector_operations_t, 
    //     KF_3D_t,
    //     lin_op_t,
    //     lin_solver_t
    //     > system_operator_t;

    typedef nonlinear_operators::system_operator_time_globalization<
        gpu_vector_operations_t, 
        KF_3D_t,
        lin_op_t,
        lin_solver_t
        > system_operator_tg_t;

    typedef numerical_algos::newton_method::newton_solver<
        gpu_vector_operations_t, 
        KF_3D_t,
        system_operator_tg_t, 
        convergence_newton_t
        > newton_t;

    log_t *log_p = &log;
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops, log_p);
    
    lin_op_t* lin_op_p = &lin_op;
    lin_solver_t* lin_solver_p = &lin_solver;
    system_operator_tg_t *system_operator_td = new system_operator_tg_t(vec_ops, lin_op_p, lin_solver_p);
    newton_t *newton = new newton_t(vec_ops, system_operator_td, conv_newton);

    conv_newton->set_convergence_constants(newton_def_tol, newton_def_max_it);

    vec_ops->assign(x_back, x0);
    bool converged = newton->solve(KF_3D, x0, Rey);
    if(!converged)
    {
        printf("Newton 2 failed to converge!\n");
    }
    printf("Newton 2 solution norm = %le, div = %le\n", vec_ops->norm(x0), KF_3D->div_norm(x0));
    KF_3D->write_solution_abs("x_2.pos", x0);


    vec_ops->stop_use_vector(b); vec_ops->free_vector(b);
    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
    vec_ops->stop_use_vector(x_back); vec_ops->free_vector(x_back);
 

    delete newton;
    delete system_operator_td;
    delete conv_newton;
    

    delete KF_3D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUFFT_C2R;
    delete CUBLAS;
    
    return 0;
}