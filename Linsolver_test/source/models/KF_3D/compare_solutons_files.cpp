#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

#include <common/macros.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/cuda_support.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>

#include <nonlinear_operators/save_norms_from_file.h>

#define Blocks_x_ 32
#define Blocks_y_ 16
using real = SCALAR_TYPE;
using complex = thrust::complex<real>;
using gpu_vector_operations_real_t = gpu_vector_operations<real>;
using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
using gpu_vector_operations_t = gpu_vector_operations<real>;
using gpu_file_operations_t = gpu_file_operations<gpu_vector_operations_t>;
using cufft_type = cufft_wrap_R2C<real>;
using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
        gpu_vector_operations_real_t, 
        gpu_vector_operations_complex_t, 
        gpu_vector_operations_t,
        Blocks_x_, Blocks_y_>;

using vec = gpu_vector_operations_t::vector_type;
using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;
using save_norms_t = nonlinear_operators::save_norms_from_file<gpu_vector_operations_t, vec_file_ops_t, KF_3D_t>;

int main (int argc, char *argv[])
{

    if(argc != 7)
    {
        std::cout << argv[0] << " nz alpha R N folder \"file_name_regex\":\n nz = 0/1(z component force), 0<alpha<=1, R - Reynolds number, N = 2^n- discretization in one direction\n folder is the path to the folder, where previously found solutions are\n \"file_name_regex\" IN QUOTES is the regular expression for the solution files  \n";
        return(0);       
    }
    int nz = std::stoi(argv[1]);
    real alpha = std::stof(argv[2]);
    real R = std::stof(argv[3]);
    size_t N = std::stoi(argv[4]);
    std::string folder_saved_solutions(argv[5]);
    std::string regex_saved_solutions(argv[6]);    
    auto solution_files = file_operations::match_file_names(folder_saved_solutions, regex_saved_solutions);
    int one_over_alpha = int(1/alpha);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;    
    size_t Mz = N/2+1;
    size_t Nv = 3*(Nx*Ny*Mz-1);
    

    init_cuda(-1);
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);    
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(Nv, CUBLAS);
    vec_file_ops_t file_ops(vec_ops);

    auto n_solution = solution_files.size();
    

    //saving bifurcation norms in a file
    {
        cufft_type cufft_c2r(Nx, Ny, Nz);
        gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, CUBLAS);
        gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, CUBLAS);
        KF_3D_t kf3d_y(alpha, Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, vec_ops, &cufft_c2r, true, nz);
        save_norms_t save_norms(vec_ops, &file_ops, &kf3d_y);
        std::stringstream ss;
        ss << "debug_curve_" << R << ".dat";
        auto save_file_name(ss.str()); 
        save_norms.save_norms_all_files(save_file_name, R, folder_saved_solutions, regex_saved_solutions);
        std::cout << "saved norms in " << save_file_name << std::endl;
    }

    std::ofstream f_corr_matrix("correlation_matrix.csv", std::ofstream::out);
    if (!f_corr_matrix) throw std::runtime_error("error while opening file \"correlation_matrix.csv\"");

        //TODO: it's better to use allocator
    std::vector<vec> vecs;
    for(auto &v: solution_files)
    {   
        vec x;
        vec_ops->init_vector(x); vec_ops->start_use_vector(x);
        file_ops.read_vector(v, (vec&)x);
        vecs.push_back(x);
        std::cout << "added data from " << v << " to storage" << std::endl;
    }
    
    std::vector< std::pair<std::string, std::string> > same_vectors; 
    std::cout << "correlation matrix: ";
    vec dd;
    vec_ops->init_vector(dd); vec_ops->start_use_vector(dd);
    for(std::size_t j = 0;j<n_solution;j++)
    {
        for(std::size_t k = 0;k<n_solution;k++)
        {
            auto x = vecs[j];
            auto y = vecs[k];
            vec_ops->assign_mul(-1.0, x, 1.0, y, dd);
            auto solution_norm = vec_ops->norm(x);
            if(solution_norm < 1.0)
            {
                solution_norm = 1.0;
            }
            auto diff_norm = vec_ops->norm(dd)/solution_norm;
            f_corr_matrix << diff_norm;
            if(k < n_solution-1)  f_corr_matrix << ",";

            if(j<=k)
            {
                if( ( diff_norm < 1.0e-4 )&&(j!=k) )
                {
                    same_vectors.push_back( {solution_files[j], solution_files[k]} );
                }
            }
        }
        f_corr_matrix << std::endl;
    }
    vec_ops->stop_use_vector(dd); vec_ops->free_vector(dd);
    std::cout << " \"correlation_matrix.csv\" done." << std::endl;

    for(auto& v: same_vectors)
    {
        std::cout << "names: (" << v.first << ")(" << v.second << ")" << std::endl;
    }
    std::cout << std::endl;

    for(auto &x: vecs)
    {
        vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
    }
    
    delete vec_ops;
    delete CUBLAS;


    return 0;
}
