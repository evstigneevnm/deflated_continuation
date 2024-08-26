#include <limits>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <common/gpu_file_operations.h>
#include <common/file_operations.h>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>
#include <common/gpu_file_operations_functions.h>
#include <test_inst/IRAM/system_operator_test.h>
#include <test_inst/IRAM/linear_operator.h>
#include <stability/IRAM/iram_process.hpp>

int main(int argc, char const *argv[])
{
    using real = SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using mat_files_t = gpu_matrix_file_operations<mat_ops_t>;
    using log_t = utils::log_std;
    using sys_op_t = stability::system_operator_stability<vec_ops_t, mat_ops_t, log_t>;
    using lin_op_t = stability::linear_operator<vec_ops_t, mat_ops_t>;
    using lapack_wrap_t = lapack_wrap<real>;
    using iram_t = stability::IRAM::iram_process<vec_ops_t, mat_ops_t, lapack_wrap_t, lin_op_t, log_t>;//, sys_op_t>;

    bool testing = false;
    if(argc == 1)
    {
        std::cout << "Usage: " << argv[0] << " matrix_file_A matrix_file_P(A) (v0_file)" << std::endl;  
        std::cout << "running in testin mode for default 'dat_files/A.dat' and 'dat_files/iP.dat' " << std::endl;
        testing = true;
    }    
    else if((argc != 3)&&(argc != 4))
    {
        std::cout << "Usage: " << argv[0] << " matrix_file_A matrix_file_P(A) (v0_file)" << std::endl;  
        return 0; 
    }

    std::string A_file_name("dat_files/A.dat");
    std::string PA_file_name("dat_files/iP.dat");
    std::string v_file_name("none");
    if(argc == 4)
    {
        v_file_name = std::string(argv[3]);
    }
    if(!testing)
    {
        A_file_name = std::string(argv[1]);
        PA_file_name = std::string(argv[2]);
    }
    auto NN = file_operations::read_matrix_size(A_file_name);
    size_t N = NN.first;
    init_cuda(-1);
    std::cout << "system size = " << N << std::endl;
    unsigned int k0 = 6;
    unsigned int m = k0*4;

    cublas_wrap CUBLAS(true);
    lapack_wrap_t lapack(m);
    log_t log;
    vec_ops_t vec_ops_N(N, &CUBLAS);
    vec_ops_t vec_ops_m(m, &CUBLAS);
    mat_ops_t mat_ops_A(N, N, &CUBLAS);
    mat_ops_t mat_ops_N(N, m, &CUBLAS);
    mat_ops_t mat_ops_m(m, m, &CUBLAS);
    vec_file_ops_t vec_file_ops(&vec_ops_N);
    mat_files_t mat_f_m(&mat_ops_m);
    mat_files_t mat_f_N(&mat_ops_N);
    mat_files_t mat_f_A(&mat_ops_A);
    
    T_mat A;
    T_mat PA;
    T_vec v0;
    mat_ops_A.init_matrix(A); mat_ops_A.start_use_matrix(A);
    mat_ops_A.init_matrix(PA); mat_ops_A.start_use_matrix(PA);
    vec_ops_N.init_vector(v0); vec_ops_N.start_use_vector(v0);
    mat_f_A.read_matrix(A_file_name, A );
    mat_f_A.read_matrix(PA_file_name, PA );
    if(v_file_name != "none")
    {
        vec_file_ops.read_vector(v_file_name, v0);
    }

    lin_op_t lin_op(&vec_ops_N, &mat_ops_A);
    sys_op_t sys_op(&vec_ops_N, &mat_ops_A, &log);
    lin_op.set_matrix_ptr(A);
    sys_op.set_matrix_ptr(PA);


    iram_t IRAM(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &lapack, &lin_op, &log);//, &sys_op);
        
    IRAM.set_target_eigs("LR");
    IRAM.set_number_of_desired_eigenvalues(k0);
    IRAM.set_tolerance(1.0e-8);
    IRAM.set_max_iterations(100);
    if(v_file_name != "none")
    {
        log.info_f("using initial vector from file %s", v_file_name.c_str() );
        IRAM.set_initial_vector(v0);
    }
    IRAM.set_verbocity(true);
    
    std::cout << "starting iram." << std::endl;
    // std::vector<T_vec> eigv_real;
    // std::vector<T_vec> eigv_imag;

    // auto eigs = IRAM.execute(eigv_real, eigv_imag);
    auto eigs = IRAM.execute();

    unsigned int k0_obtained = eigs.size();
    // std::cout << std::scientific;
    for(auto &e: eigs)
    {
        auto e_real = e.real();
        auto e_imag = e.imag();
        if( std::abs(e_imag) < std::numeric_limits<real>::epsilon() )
            log.info_f("%le",e_real);
        else if(e_imag>0.0)
            log.info_f("%le+%le",e_real,e_imag);
        else
            log.info_f("%le%le",e_real,e_imag);
        // std::cout << e << std::endl;
    }

    if(testing)
    {
        //reference first 10 LR values
        decltype(eigs) ref_eigs = 
        {
            {61.65033990538935,0},
            {60.69448617862329,1.386007408795112},
            {60.69448617862329,-1.386007408795112},
            {60.227813877994684,2.350252930708263},
            {60.227813877994684,-2.350252930708263},
            {60.1031534800608,+7.770486796895777},
            {60.1031534800608,-7.770486796895777},
            {59.782528167140384,+3.6899204030892423},
            {59.782528167140384,-3.6899204030892423},
            {59.67160212284957,+0.4537965010096829},
            {59.67160212284957,-0.4537965010096829}
        };
        
        std::vector<real> diff;
        for(int j=0;j<k0_obtained;j++)
        {
            auto ref = ref_eigs[j];
            auto ref2 = ref_eigs[j+1];
            auto e = eigs[j];
            // std::cout << ref << "  " << e << std::endl;
            if(std::abs(ref.imag()) > 1.0e-12 )
            {
                auto v1 = std::abs(e - ref); 
                auto v2 = std::abs(e - ref2);
                diff.push_back( std::min(v1,v2) );
            }
            else
            {
                diff.push_back( std::abs(e - ref) );
            }
        }
        log.info("error vs reference:");
        for(auto &e: diff)
        {
            log.info_f("%le", e);
        }
        auto ref_diff = std::accumulate(diff.cbegin(), diff.cend(), 0);

        vec_ops_N.stop_use_vector(v0); vec_ops_N.free_vector(v0);
        mat_ops_A.stop_use_matrix(PA); mat_ops_A.free_matrix(PA);
        mat_ops_A.stop_use_matrix(A); mat_ops_A.free_matrix(A);

        if(ref_diff < 1.0e-8)
        {
            log.info("PASS");
            return 0;
        }
        else
        {
            log.info("FAILED");
            return 1;
        }
    }
    else
    {
        return 0;
    }
}