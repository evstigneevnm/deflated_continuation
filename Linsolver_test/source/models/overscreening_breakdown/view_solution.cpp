#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

#include <utils/init_cuda.h>
#include <external_libraries/cublas_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown_params.h>
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown.h>

#include <nonlinear_operators/overscreening_breakdown/system_operator.h>
#include <nonlinear_operators/overscreening_breakdown/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>

std::string get_file_name(const std::string& filename)
{
    size_t lastslash = filename.find_last_of("/");
    size_t lastdot = filename.find_last_of(".");
        
    if ((lastdot == std::string::npos)&&(lastslash == std::string::npos))
    {
        return filename;
    }
    else
    {
        if (lastslash != std::string::npos)
        {
            if(lastslash<lastdot)
            {
                return filename.substr(lastslash+1, lastdot-lastslash-1);
            }
            else
            {
                return filename.substr(lastslash+1, filename.length() );
            }
            
        }
        else
        {
            return filename.substr(0, lastdot);
        }
    }
    
}


int main(int argc, char const *argv[])
{
    using real =  SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t>;
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;


    if(argc != 3)
    {
        std::cout << argv[0] << " L file_name" << std::endl;
        std::cout << " L>0 - mapping value, file_name - is the file that contins solution coefficients." << std::endl;
        std::cout << " DOF are deduced from the file."  << std::endl;
        std::cout << " The rest parameters are set as default because they have no influence on the output." << std::endl;
        return(0);       
    }
    
    using params_st = params_s<real>;    
    
    std::string file_name(argv[2]);
    params_st params;
    params.L = std::stof(argv[1]);
    size_t N = file_operations::read_vector_size(file_name);
    params.N = N;
    params.print_data();

    utils::init_cuda(-1);

    cublas_wrap cublas(true);
    vec_ops_t vec_ops(N, &cublas);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), vec_ops.get_cublas_ref() );
    vec_file_ops_t vec_file_ops(&vec_ops);
    
    ob_prob_t ob_prob(&vec_ops, &mat_ops, params );

    T_vec x0, x1, dx, x_back, b;

    vec_ops.init_vectors(b, x0, x1, dx, x_back); vec_ops.start_use_vectors(b, x0, x1, dx, x_back);

    vec_file_ops.read_vector(file_name, x0);    
    ob_prob.physical_solution(x0, x1);
    //fix the boundary condition on the 3-d derivative for visualization
    T_vec x_host = vec_ops.view(x1);
    x_host[N-1] = x_host[N-2];
    vec_ops.set(x1);

    std::stringstream ss;
    ss << "view_" << get_file_name(file_name) << ".dat";
    vec_file_ops.write_vector(ss.str(), x1);


    printf("||x|| = %le \n", vec_ops.norm(x0));


    vec_ops.stop_use_vectors(b, x0, x1, dx, x_back); vec_ops.free_vectors(b, x0, x1, dx, x_back);

    
    return 0;
}