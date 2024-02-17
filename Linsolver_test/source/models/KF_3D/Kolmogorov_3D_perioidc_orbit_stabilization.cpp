#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

#include <utils/init_cuda.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <periodic_orbit/periodic_orbit_nonlinear_operator.h>
#include <periodic_orbit/system_operator_single_section.h>
#include <periodic_orbit/convergence_strategy_single_section.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <stability/IRAM/iram_process.hpp>


int main(int argc, char const *argv[])
{
    const int Blocks_x_ = 64;
    const int Blocks_y_ = 16;
    
    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;
    using gpu_vector_operations_real_t = gpu_vector_operations<real>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, vec_t>;
    using mat_t = typename mat_ops_t::matrix_type;

    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;

    using cufft_type = cufft_wrap_R2C<real>;

    using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            vec_ops_t,
            Blocks_x_, Blocks_y_>;            


    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;

    using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<vec_ops_t, KF_3D_t, log_t, time_steppers::time_step_adaptation_constant, time_steppers::explicit_time_step>; //time_step_adaptation_constant //time_step_adaptation_error_control

    using periodic_orbit_linear_operator_t = typename periodic_orbit_nonlinear_operator_t::linear_operator_type;
    using periodic_orbit_preconditioner_t = typename periodic_orbit_nonlinear_operator_t::preconditioner_type;

    using lin_solve_t = numerical_algos::lin_solvers::bicgstabl<periodic_orbit_linear_operator_t, periodic_orbit_preconditioner_t, vec_ops_t, monitor_t, log_t>;

    using system_operator_single_section_t = nonlinear_operators::newton_method::system_operator_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, periodic_orbit_linear_operator_t, lin_solve_t>;    
    using convergence_strategy_single_section_t = nonlinear_operators::newton_method::convergence_strategy_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, log_t>;
    using newton_solver_t = numerical_algos::newton_method::newton_solver<vec_ops_t, periodic_orbit_nonlinear_operator_t, system_operator_single_section_t, convergence_strategy_single_section_t>;

    using lapack_wrap_t = lapack_wrap<real>;
    using iram_t = stability::IRAM::iram_process<vec_ops_t, mat_ops_t, lapack_wrap_t, periodic_orbit_linear_operator_t, log_t>;


    if((argc < 7)||(argc > 9))
    {
        std::cout << argv[0] << " cuda_dev_num N alpha R time method state_file(optional) desired_eignevalues(optional)\n  R -- Reynolds number,\n   0<alpha<=1 - torus stretching parameter,\n  N = 2^n- discretization in one direction. \n  time - simmulation time. \n";
         std::cout << " method - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    //initial timestep:
    real initial_dt = 7.0e-3;

    int cuda_dev_num = std::stoi(argv[1]);
    size_t N = std::stoul(argv[2]);
    real alpha = std::stof(argv[3]);
    int one_over_alpha = static_cast<int>(1.0/alpha);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;

    real R = std::stof(argv[4]);
    real simulation_time = std::stof(argv[5]);
    std::string scheme_name(argv[6]);
    bool load_file = false;
    std::string load_file_name;
        
    //number of desired and total eigenvalues
    size_t m = 30;
    size_t k = 0;

    std::cout << "input parameters: " << "\nN = " << N << "\none_over_alpha = " << one_over_alpha << "\nNx = " << Nx << " Ny = " << Ny << " Nz = " << Nz << "\nR = " << R << "\nsimulation_time = " << simulation_time << "\nscheme_name = " << scheme_name << "\ncuda_dev_num = " << cuda_dev_num;

    if(argc == 8)
    {
        load_file = true;
        load_file_name = std::string(argv[7]);
        std::cout << "\nload_file_name = " << load_file_name;
    }
    if(argc == 9)
    {
        load_file = true;
        load_file_name = std::string(argv[7]);
        k = std::stoul(argv[8]);
        m = k*5;
        std::cout << "\nload_file_name = " << load_file_name;
        std::cout << "\nusing eigensolver with "  << " k = " << k << " m = " << m;
    }    
    std::cout  << std::endl;

    auto method = time_steppers::detail::methods::EXPLICIT_EULER;
    if(scheme_name == "EE")
    {
        method = time_steppers::detail::methods::EXPLICIT_EULER;
    }
    else if(scheme_name == "RKDP45")
    {
        method = time_steppers::detail::methods::RKDP45;
    }
    else if(scheme_name == "RK33SSP")
    {
        method = time_steppers::detail::methods::RK33SSP;
    }    
    else if(scheme_name == "RK43SSP")
    {
        method = time_steppers::detail::methods::RK43SSP;
    } 
    else if(scheme_name == "RK64SSP")
    {
        method = time_steppers::detail::methods::RK64SSP;
    }     
    else if(scheme_name == "HE")
    {
        method = time_steppers::detail::methods::HEUN_EULER;
    }  
    else
    {
        throw std::logic_error("incorrect method string type provided.");
    }

    if( !utils::init_cuda(cuda_dev_num) )
    {
        return 2;
    } 
    cufft_type cufft_c2r(Nx, Ny, Nz);
    size_t Mz = cufft_c2r.get_reduced_size();
    cublas_wrap cublas(true);
    lapack_wrap_t lapack(m);
    cublas.set_pointer_location_device(false);

    log_t log;
    log_t log_ls;
    log_ls.set_verbosity(0);
    if(k>0)
    {
        log.info_f("using eigensolver with m = %i and k = %i", m, k);
    }
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    vec_ops_t vec_ops(3*(Nx*Ny*Mz-1), &cublas);
    vec_file_ops_t file_ops(&vec_ops);

 
    vec_t x0,x1;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);
    vec_ops.init_vector(x1); vec_ops.start_use_vector(x1);

    KF_3D_t KF3D(alpha, Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        KF3D.randomize_vector(x0);
    }
    periodic_orbit_nonlinear_operator_t periodic_orbit_nonlin_op(&vec_ops, &KF3D, &log, 200.0, R, method, initial_dt);

    auto periodic_orbit_lin_op = periodic_orbit_nonlin_op.linear_operator;

    lin_solve_t solver(&vec_ops, &log, 0);
    solver.set_basis_size(4);
    real rel_tol = 1.0e-2;
    size_t max_iters = 15;
    auto& mon = solver.monitor();
    mon.init(rel_tol, real(0), max_iters);
    mon.set_save_convergence_history(true);
    mon.set_divide_out_norms_by_rel_base(true);    

    real newton_tol = 1.0e-9;
    newton_tol *= vec_ops.get_l2_size();
    system_operator_single_section_t sys_op(&vec_ops, periodic_orbit_lin_op, &solver);
    convergence_strategy_single_section_t convergence(&vec_ops, &log, newton_tol);
    newton_solver_t newton(&vec_ops, &sys_op, &convergence);
    periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses(x0, R);
    if(simulation_time>0.0)
    {
        std::stringstream ss;
        ss << "KF3D_initial_R_" << R << ".dat";
        periodic_orbit_nonlin_op.time_stepper(x0, R, {0, simulation_time});
        periodic_orbit_nonlin_op.save_norms( ss.str() );

        std::stringstream ss_t;
        ss_t << "x_" << simulation_time << "_R_" << R << ".dat";
        file_ops.write_vector(ss_t.str(), x0);
    }
    bool is_newton_converged = newton.solve(&periodic_orbit_nonlin_op, x0, R);
    
    if(is_newton_converged)
    {
        std::stringstream ss_periodic_estimate;
        ss_periodic_estimate << "KF3D_period_R_" << R << "_scheme_" << scheme_name << ".dat";
        periodic_orbit_nonlin_op.save_period_estmate_norms(ss_periodic_estimate.str() );

        std::stringstream ss, ss1, sseigs;
        ss << "x_" << simulation_time << "_R_" << R << "_periodic.pos";
        auto pos_file_name(ss.str());
        KF3D.write_solution_abs(pos_file_name, x0);     

        ss1 << "x_" << simulation_time << "_R_" << R << "_periodic.dat";
        file_ops.write_vector(ss1.str(), x0);

        if(k>0)
        {
            
            vec_ops_t vec_ops_m(m, &cublas);
            mat_ops_t mat_ops_N(N, m, &cublas);
            mat_ops_t mat_ops_m(m, m, &cublas);
            //setting up eigenvalue problem
            iram_t iram(&vec_ops, &mat_ops_N, &vec_ops_m, &mat_ops_m, &lapack, periodic_orbit_lin_op, &log);
            log.info("=== starting up eigenvalue problem ===");
            iram.set_target_eigs("LM");
            iram.set_number_of_desired_eigenvalues(k);
            iram.set_tolerance(newton_tol);
            iram.set_max_iterations(100);
            iram.set_verbocity(true);
            KF3D.randomize_vector(x1, 18);
            iram.set_initial_vector(x1);
            
            std::vector<vec_t> eigvs_real;
            std::vector<vec_t> eigvs_imag;

            for(int jj=0;jj<3;jj++)
            {
                vec_t x_r,x_i;
                vec_ops.init_vector(x_r); vec_ops.start_use_vector(x_r); 
                vec_ops.init_vector(x_i); vec_ops.start_use_vector(x_i); 
                eigvs_real.push_back(x_r);
                eigvs_imag.push_back(x_i);
            }


            auto eigs = iram.execute(eigvs_real, eigvs_imag);
            

            for(auto &e: eigs)
            {
                auto e_real = e.real();
                auto e_imag = e.imag();
                if( std::abs(e_imag) < std::numeric_limits<real>::epsilon() )
                    log.info_f("%le",1.0+e_real); //shift for Monodromy matrix
                else if(e_imag>0.0)
                    log.info_f("%le+%le",1.0+e_real,e_imag);
                else
                    log.info_f("%le%le",1.0+e_real,e_imag);
                // std::cout << e << std::endl;

            }
            sseigs << "KF3D_eigs_" << simulation_time << "_R_" << R << "_periodic.dat";
            std::ofstream f(sseigs.str(), std::ofstream::out);
            if (!f) throw std::runtime_error("error while opening file " + sseigs.str());

            for(auto &e: eigs)
            {
                if (!(f << std::scientific << std::setprecision(17) << 1.0+e.real() << "," << e.imag() <<  std::endl))
                    throw std::runtime_error("error while writing to file " + sseigs.str());
            }        
            f.close();         

            size_t jj = 0;
            for(auto &x: eigvs_real)
            {
                std::stringstream ss_eigvs_real, sv_eigvs_real;
                ss_eigvs_real << "KF3D_eigv_real_" << jj << "_" << simulation_time << "_R_" << R << "_periodic.pos";
                KF3D.write_solution_abs(ss_eigvs_real.str(), x); 
                sv_eigvs_real << "KF3D_eigv_real_" << jj << "_" << simulation_time << "_R_" << R << "_periodic.dat";
                file_ops.write_vector(sv_eigvs_real.str(), x);
                ++jj;
            }
            jj = 0;
            for(auto &x: eigvs_imag)
            {
                std::stringstream ss_eigvs_imag, sv_eigvs_imag;
                ss_eigvs_imag << "KF3D_eigv_imag_" << jj << "_" << simulation_time << "_R_" << R << "_periodic.pos";
                KF3D.write_solution_abs(ss_eigvs_imag.str(), x); 
                sv_eigvs_imag << "KF3D_eigv_imag_" << jj << "_" << simulation_time << "_R_" << R << "_periodic.dat";
                file_ops.write_vector(sv_eigvs_imag.str(), x);   
                ++jj;         
            }

            for(int jj=0;jj<3;jj++)
            {
                vec_ops.stop_use_vector(eigvs_real.at(jj)); vec_ops.free_vector(eigvs_real.at(jj)); 
                vec_ops.stop_use_vector(eigvs_imag.at(jj)); vec_ops.free_vector(eigvs_imag.at(jj));
            }

        }
    }

    vec_ops.stop_use_vector(x1); vec_ops.free_vector(x1);
    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);

    return 0;
}