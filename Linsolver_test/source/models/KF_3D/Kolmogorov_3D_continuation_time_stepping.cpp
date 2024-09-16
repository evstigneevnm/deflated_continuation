#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <memory>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

#include <scfd/utils/init_cuda.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <scfd/utils/log.h>
#include <scfd/utils/cuda_timer_event.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>




#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D_stiff.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D_stiff.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D_shifted.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D_shifted.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>

#include <time_stepper/detail/butcher_tables.h>

#include <time_stepper/time_step_adaptation_tolerance.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/generic_time_step.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/explicit_implicit_time_step.h>
#include <time_stepper/implicit_time_step.h>
#include <time_stepper/distance_to_points.h>
#include <time_stepper/time_stepper.h>





int main(int argc, char const *argv[])
{
    
    const int Blocks_x_ = 32;
    const int Blocks_y_ = 16;

    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;
    using gpu_vector_operations_real_t = gpu_vector_operations<real>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using gpu_vector_operations_t = gpu_vector_operations<real>;
    using cufft_type = cufft_wrap_R2C<real>;
    using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_>;     

    using linear_operator_K_3D_stiff_t = nonlinear_operators::linear_operator_K_3D_stiff<gpu_vector_operations_t, KF_3D_t>;
    using preconditioner_K_3D_stiff_t = nonlinear_operators::preconditioner_K_3D_stiff<gpu_vector_operations_t, KF_3D_t, linear_operator_K_3D_stiff_t>;
    using linear_operator_K_3D_temp_t = nonlinear_operators::linear_operator_K_3D_shifted<gpu_vector_operations_t, KF_3D_t>;
    using preconditioner_K_3D_temp_t = nonlinear_operators::preconditioner_K_3D_shifted<gpu_vector_operations_t, KF_3D_t, linear_operator_K_3D_temp_t>;

    using lin_op_stiff_t = linear_operator_K_3D_stiff_t;
    using prec_stiff_t = preconditioner_K_3D_stiff_t;
    
    using lin_op_t = linear_operator_K_3D_temp_t;
    using prec_t = preconditioner_K_3D_temp_t;

    using real_vec_t = typename gpu_vector_operations_real_t::vector_type; 
    using complex_vec_t = typename gpu_vector_operations_complex_t::vector_type;
    using vec_t = typename gpu_vector_operations_t::vector_type;

    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;


    using log_t = scfd::utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<gpu_vector_operations_t,log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<gpu_vector_operations_t, log_t>;
    using time_step_adaptation_constant_t = time_steppers::time_step_adaptation_constant<gpu_vector_operations_t, log_t>;
    using time_step_adaptation_tolerance_t = time_steppers::time_step_adaptation_tolerance<gpu_vector_operations_t, log_t>;

    using lin_solver_stiff_t = numerical_algos::lin_solvers::bicgstabl<lin_op_stiff_t,prec_stiff_t,gpu_vector_operations_t,monitor_t,log_t>;
    using lin_solver_t = numerical_algos::lin_solvers::bicgstabl<lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;
    using exact_solver_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_stiff_t,prec_stiff_t,gpu_vector_operations_t,monitor_t,log_t>;

    using time_step_explicit_t = time_steppers::explicit_time_step<gpu_vector_operations_t, KF_3D_t, log_t, time_step_err_ctrl_t>; //time_step_err_ctrl_t time_step_adaptation_tolerance_t time_step_adaptation_constant_t
    using time_step_imex_t = time_steppers::explicit_implicit_time_step<gpu_vector_operations_t, KF_3D_t, lin_op_stiff_t, exact_solver_t, log_t, time_step_adaptation_constant_t>; 
    using time_step_implicit_t = time_steppers::implicit_time_step<gpu_vector_operations_t, KF_3D_t, lin_op_t, lin_solver_t,log_t, time_step_adaptation_tolerance_t>; //time_step_adaptation_constant_t  time_step_err_ctrl_t time_step_adaptation_tolerance_t

    using time_step_generic_t = time_steppers::generic_time_step<gpu_vector_operations_t>;

    // using time_stepper_explicit_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_explicit_t, log_t>;
    // using time_stepper_imex_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_imex_t, log_t>;    
    // using time_stepper_implicit_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_implicit_t, log_t>;

    using distance_to_points_t = time_steppers::distance_to_points<gpu_vector_operations_t>;
    using time_stepper_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_generic_t, log_t, distance_to_points_t>;


    using timer_t = scfd::utils::cuda_timer_event;

    if((argc < 7)||(argc > 10))
    {
        
        std::cout << argv[0] << " N alpha R time method GPU_PCI_ID [state_file folder \"file_name_regex\"](optional)\n   0<alpha<=1 - torus stretching parameter,\n R -- Reynolds number,\n  N = 2^n- discretization in one direction. \n  time - simulation time. \n  folder - for the points daata";
        std::cout << " method - name of the scheme:" << std::endl;

        time_steppers::detail::butcher_tables tables;
        time_steppers::detail::composite_butcher_tables composite_tables;

        auto names = tables.get_list_of_table_names();
        auto composite_names = composite_tables.get_list_of_table_names();
        for(auto &n: names)
        {
            std::cout << n << " ";
        }
        for(auto &n: composite_names)
        {
            std::cout << n << " ";
        }        
        std::cout << std::endl;
        return(0);       
    }    
    real initial_dt = 1.0e-15;
    size_t N = std::stoul(argv[1]);
    real alpha = std::stof(argv[2]);
    int one_over_alpha = static_cast<int>(1.0/alpha);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    int nz = 1;
    real R = std::stof(argv[3]);
    real simulation_time = std::stof(argv[4]);
    std::string scheme_name(argv[5]);
    int gpu_pci_id = std::stoi(argv[6]);
    bool load_file = false;
    std::string load_file_name;
    if((argc == 8)||(argc == 10))
    {
        load_file = true;
        load_file_name = std::string(argv[7]);
    }
    std::string folder_saved_solutions;
    std::string regex_saved_solutions; 
    if(argc == 10)
    {
        folder_saved_solutions = std::string(argv[8]);
        regex_saved_solutions = std::string(argv[9]);
    }


    std::cout << "input parameters: " << "\nN = " << N << "\none_over_alpha = " << one_over_alpha << "\nNx = " << Nx << " Ny = " << Ny << " Nz = " << Nz << "\nR = " << R << "\nsimulation_time = " << simulation_time << "\nscheme_name = " << scheme_name << "\ngpu_pci_id = " << gpu_pci_id << "\n";
//  sqrt(L*scale_force)/(n R):
    const real scale_force = 1.0;
    std::cout << "reduced Reynolds number = " <<  std::sqrt(one_over_alpha*2*3.141592653589793238*scale_force)*R << std::endl;
    if(load_file)
    {
        std::cout << "\nload_file_name = " << load_file_name;
    }
    std::cout  << std::endl;

    // get scheme type from scheme name
    auto scheme_type = time_steppers::get_scheme_type_by_name(scheme_name);
    std::cout << "scheme type = " << scheme_type << std::endl;
    std::cout << "======= X =======" << std::endl;
    // if scheme name is incorrect, then the runtime error will be thrown

    scfd::utils::init_cuda(gpu_pci_id);
    cufft_type cufft_c2r(Nx, Ny, Nz);
    size_t Mz = cufft_c2r.get_reduced_size();
    cublas_wrap cublas(true);
    cublas.set_pointer_location_device(false);
    log_t log;
    log_t log_ls;
    log_ls.set_verbosity(0);
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    size_t N_global = 3*(Nx*Ny*Mz-1);
    gpu_vector_operations_t vec_ops(N_global, &cublas);
    
    distance_to_points_t distance_to_points(&vec_ops);

    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0, x1;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);

    KF_3D_t kf3d_y(alpha, Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r, true, nz);

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        vec_ops.init_vector(x1); vec_ops.start_use_vector(x1);
        kf3d_y.randomize_vector(x1);
        kf3d_y.exact_solution(R, x0);
        vec_ops.add_mul(-0.5, x1, x0);
        vec_ops.stop_use_vector(x1); vec_ops.free_vector(x1);
    }
    if(!folder_saved_solutions.empty())
    {
        auto solution_files = file_operations::match_file_names(folder_saved_solutions, regex_saved_solutions);
        vec_ops.init_vector(x1); vec_ops.start_use_vector(x1);
        for(auto &v: solution_files)
        {   
            file_ops.read_vector(v, (vec_t&)x1);
            distance_to_points.copy_and_add(x1);
            std::cout << "added data from " << v << " to storage" << std::endl;
        }  
        vec_ops.stop_use_vector(x1); vec_ops.free_vector(x1);
    }

    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log, {0.0, simulation_time});
    time_step_adaptation_constant_t time_step_const(&vec_ops, &log, {0.0, simulation_time}, initial_dt);
    time_step_adaptation_tolerance_t time_step_tol(&vec_ops, &log, {0.0, simulation_time}, initial_dt);

    time_step_tol.set_adaptation_method("H321", 3); //"I",3 //addaptiation type, ode solver order
    time_step_tol.set_parameters(1.0e-5);


    
    
    //linsolver control
    unsigned int lin_solver_max_it = 200;
    real lin_solver_tol = 1.0e-3;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;

    lin_op_stiff_t lin_op_stiff(&vec_ops, &kf3d_y);
    lin_op_t lin_op(&vec_ops, &kf3d_y);
    prec_stiff_t prec_stiff(&vec_ops, &kf3d_y);
    prec_t prec(&kf3d_y);

    log_t log3;
    log3.set_verbosity(0);    
    exact_solver_t exact_solver(&vec_ops, &log3);
    exact_solver.set_preconditioner(&prec_stiff);

    lin_solver_t lin_solver(&vec_ops, &log3); 
    lin_solver.set_preconditioner(&prec);
    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver.set_basis_size(basis_sz);
    
    auto mon = &lin_solver.monitor();
    mon->init(lin_solver_tol, real(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);    


    if (scheme_name == "IE")
    {
        time_step_tol.force_globalization();
    }

    std::shared_ptr<time_step_explicit_t> explicit_step;
    std::shared_ptr<time_step_implicit_t> implicit_step;
    std::shared_ptr<time_stepper_t> time_stepper;



    if(scheme_type == "implicit")
    {
        implicit_step = std::make_shared<time_step_implicit_t>(&vec_ops, &time_step_tol, &log3, &kf3d_y, &lin_op, &lin_solver, R, scheme_name); //& time_step_err_ctrl
        implicit_step->set_newton_method(1.0e-9, 100);        
        time_stepper = std::make_shared<time_stepper_t>(&vec_ops, &kf3d_y, implicit_step.get(), &log, &distance_to_points);
    }
    else if (scheme_type == "explicit")
    {
        
        explicit_step = std::make_shared<time_step_explicit_t>(&vec_ops, &time_step_err_ctrl, &log, &kf3d_y, R, scheme_name);
        time_stepper = std::make_shared<time_stepper_t>(&vec_ops, &kf3d_y, explicit_step.get(), &log, &distance_to_points);
    }
    else if (scheme_type == "imex")
    {
        time_step_imex_t imex_step(&vec_ops, &time_step_const, &log, &kf3d_y, &lin_op_stiff, &exact_solver, R, scheme_name );
        time_stepper = std::make_shared<time_stepper_t>(&vec_ops, &kf3d_y, &imex_step, &log, &distance_to_points);
    }



    time_stepper->set_skip(100);

    log.info_f("executing time stepper with time = %.2le", simulation_time);

    time_stepper->set_parameter(R);
    // time_stepper.set_time(simulation_time);
    time_stepper->set_initial_conditions(x0);
    // time_stepper.get_results(x0);
    // time_stepper.set_skip(1000);
    timer_t e1,e2;
    e1.record();
    time_stepper->execute();
    e2.record();
    std::cout << "elapsed_time = " << e2.elapsed_time(e1) << " ms" << std::endl;

    time_stepper->save_norms("probe_1.dat");
    time_stepper->get_results(x0);

    distance_to_points.save_results("distances.dat");



    std::stringstream ss, ss1;
    ss << "x_" << simulation_time << "_sim.pos";
    auto pos_file_name(ss.str());
    kf3d_y.write_solution_abs(pos_file_name, x0);     

    ss1 << "x_" << simulation_time << "_R_" << R << "_res.dat";
    file_ops.write_vector(ss1.str(), x0);


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

