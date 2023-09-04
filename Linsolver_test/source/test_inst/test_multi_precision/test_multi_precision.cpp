#include <boost/multiprecision/cpp_bin_float.hpp>
#include <iostream>
#include <common/cpu_vector_operations_var_prec.h>
#include <common/file_operations.h>
#include <common/cpu_matrix_vector_operations_var_prec.h>
#include <common/cpu_matrix_file_operations.h>
#include <contrib/scfd/include/scfd/utils/device_tag.h>
#include <contrib/scfd/include/scfd/utils/system_timer_event.h>

//struct under external managment!
template<class T, class MatrixType>
struct matrix_s
{
    matrix_s(size_t rows_p, size_t cols_p):
    rows(rows_p), cols(cols_p)
    {}

    __DEVICE_TAG__ inline T& operator()(size_t j, size_t k)
    {
        // return data[cols*j+k];
        return data[rows*k+j];
    }
    size_t rows;
    size_t cols;
    MatrixType data;        
};

int main(int argc, char const *argv[]) 
{
    if(argc != 2)
    {
        std::cout << "usage: " << argv[0] << " size" << std::endl;
        return 0;
    }

    size_t N = std::stoul(argv[1]);

    using T_100 = boost::multiprecision::cpp_bin_float_100;
    T_100 a = 2;
    T_100 b = 10;
    T_100 c = 10;

    // d = a * b + c
    auto d = boost::multiprecision::fma(a,b,c);
    std::cout << "fma:" << a << "*" << b << "+" << c << "=" << d << std::endl;
    

    using vec_ops_t = cpu_vector_operations_var_prec<100>;
    using T = typename vec_ops_t::scalar_type;
    using mat_ops_t = cpu_matrix_vector_operations_var_prec<vec_ops_t>;
    using mat_data_t = typename mat_ops_t::matrix_type;
    using matrix_t = matrix_s<T, mat_data_t>;
    using vec_t = typename vec_ops_t::vector_type;
    using timer_t = scfd::utils::system_timer_event;
    using mat_files_t = cpu_matrix_file_operations<mat_ops_t>;

    T a200 = 2;

    std::cout << std::setprecision(200) << std::endl;

    std::cout << boost::multiprecision::sin(a200) << std::endl;
    std::cout << boost::multiprecision::sinh(a200) << std::endl;
    std::cout << boost::multiprecision::cos(a200) << std::endl;
    std::cout << boost::multiprecision::cosh(a200) << std::endl;
    std::cout << boost::multiprecision::tan(a200) << std::endl;
    std::cout << boost::multiprecision::exp(a200) << std::endl;
    std::cout << boost::multiprecision::log(b) << std::endl;
    std::cout << boost::multiprecision::sqrt(b) << std::endl;

    std::cout << "cmp float_100 and float_200" << std::endl; 
    std::cout << std::setprecision(120) << std::endl;
    std::cout << boost::multiprecision::sinh(a) << std::endl;
    std::cout << boost::multiprecision::sinh(a200) << std::endl;
    std::cout << std::setprecision(17) << std::endl;
    std::cout << "diff = " << boost::multiprecision::sinh(a200) - boost::multiprecision::sinh(a) << std::endl;
    std::cout << "is inf? = " << boost::multiprecision::isinf(a) << std::endl;
    //impement tests for vector and matrix operations

    vec_ops_t vec_ops(N);
    vec_t x, y, z, r, rhs, sol;
    vec_ops.init_vectors(x,y,z,r,rhs, sol); vec_ops.start_use_vectors(x,y,z,r,rhs, sol);
    vec_ops.assign_scalar(1, x);
    vec_ops.assign_scalar(boost::multiprecision::sqrt(static_cast<T>(2.0)), y);

    std::cout << " norm(x) = " << vec_ops.norm(x) << std::endl;
    std::cout << " norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.normalize(y);
    std::cout << " normalized y norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.add_mul_scalar(static_cast<T>(2.0), boost::multiprecision::sqrt(static_cast<T>(2.0)), y);
    std::cout << " add mul scalar y norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.assign(y, z);
    std::cout << " assigned y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    std::cout << " is result valid?: " << vec_ops.check_is_valid_number(z) << std::endl;
    vec_ops.assign_mul(static_cast<T>(3.0), y, z);
    std::cout << " assign_mul 3y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    vec_ops.assign_mul(static_cast<T>(6.0), x, static_cast<T>(-3.0), y, z);
    std::cout << " assign_mul 6x-3y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    vec_ops.set_value_at_point(static_cast<T>(100.0), 1, z);
    std::cout << " z[55] = 100.0 == " << vec_ops.get_value_at_point(1, z) << std::endl;
    std::cout << " max(z) = " << vec_ops.max_element(z) << std::endl;
    std::cout << " argmax(z) = " << vec_ops.argmax_element(z) << std::endl;


    vec_ops.assign_random(r);
    std::cout << " random r norm(r) = " << vec_ops.norm(r) << std::endl;
    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << r[j] << " ";
    }
    std::cout << std::endl;
    vec_ops.assign_random(r);
    std::cout << " random r norm(r) = " << vec_ops.norm(r) << std::endl;
    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << r[j] << " ";
    }
    std::cout << std::endl;
    vec_ops.assign_random(r,static_cast<T>(-100.0),static_cast<T>(100.0) );
    std::cout << " random r norm(r) = " << vec_ops.norm(r) << std::endl;
    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << r[j] << " ";
    }
    std::cout << std::endl;

    mat_ops_t mat_ops(N, N, &vec_ops);
    mat_files_t mat_files(&mat_ops);

    matrix_t P(N,N), A(N,N), A_save(N,N), B(N,N), C(N,N), D(N,N), D1(N,N), DTN(N,N), DNT(N,N), C_geam(N,N), C_gemm(N,N), C1_gemm(N,N), CTN_gemm(N,N), CNT_gemm(N,N), iA_ref(N,N), iA(N,N);

    mat_ops.init_matrices(P.data, A_save.data, A.data, B.data, C.data, D.data, D1.data, DTN.data, DNT.data, C_geam.data, C_gemm.data, C1_gemm.data, CTN_gemm.data, CNT_gemm.data, iA_ref.data, iA.data); 
    mat_ops.start_use_matrices(P.data, A_save.data, A.data, B.data, C.data, D.data, D1.data, DTN.data, DNT.data, C_geam.data, C_gemm.data, C1_gemm.data, CTN_gemm.data, CNT_gemm.data, iA_ref.data, iA.data);

    bool all_files_read = false;
    try
    {
        auto sizes = mat_files.read_matrix_size("A.dat");
        if(sizes.first != N)
        {
            throw std::runtime_error("matrix sizes don't match, matrix size from file is: " + std::to_string(sizes.first) + " X " + std::to_string(sizes.second) );
        }
        mat_files.read_matrix("A.dat", A.data);
        mat_files.read_matrix("B.dat", B.data);
        mat_files.read_matrix("C_geam.dat", C_geam.data);
        mat_files.read_matrix("C_gemm.dat", C_gemm.data);
        mat_files.read_matrix("C1_gemm.dat", C1_gemm.data);
        mat_files.read_matrix("CTN_gemm.dat", CTN_gemm.data);
        mat_files.read_matrix("CNT_gemm.dat", CNT_gemm.data);
        file_operations::read_vector<T, vec_t>("b.dat",  N, rhs);
        file_operations::read_vector<T, vec_t>("x.dat",  N, sol);
        mat_files.read_matrix("iA.dat", iA_ref.data);

        all_files_read = true;
    }
    catch(const std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Using random data." << std::endl;
        mat_ops.assign_random(A.data);
        mat_ops.assign_random(B.data);        
        vec_ops.assign_random(rhs);  
    }

    timer_t matrix_event_1, matrix_event_2;
    matrix_event_1.record();



    mat_ops.gemv('N', A.data, static_cast<T>(1.0), x, static_cast<T>(0.0), y);
    mat_ops.gemv('T', A.data, static_cast<T>(1.0), x, static_cast<T>(0.0), z);
    mat_ops.mat2column_dot_vec(A.data, 3, static_cast<T>(1.0), x, static_cast<T>(0.0), r);

    mat_ops.geam('N', N, N, static_cast<T>(1.0), A.data, static_cast<T>(1.0), B.data, C.data);

    mat_ops.gemm('N', 'N', static_cast<T>(1.0), N, N, A.data, N, N, N, B.data, static_cast<T>(0.0), D.data);
    mat_ops.gemm('T', 'N', static_cast<T>(1.0), N, N, A.data, N, N, N, B.data, static_cast<T>(0.0), DTN.data);
    mat_ops.gemm('N', 'T', static_cast<T>(1.0), N, N, A.data, N, N, N, B.data, static_cast<T>(0.0), DNT.data);
    mat_ops.assign(D.data, D1.data);
    mat_ops.gemm('N', 'N', static_cast<T>(1.0), N, N, A.data, N, N, N, B.data, static_cast<T>(1.0), D1.data);
    matrix_event_2.record();

    mat_ops.assign(A.data, iA.data);
    mat_ops.lup_decomposition(iA.data, P.data);
    mat_files.write_matrix("LU_b.dat", iA.data, 200);
    mat_files.write_matrix("P_b.dat", P.data, 1);

    timer_t solve_s, solve_e;
    mat_ops.assign(A.data, iA.data);
    solve_s.record();
    mat_ops.gesv(iA.data, rhs, x);
    solve_e.record();

    timer_t inv_s, inv_e;
    mat_ops.assign(A.data, A_save.data);
    inv_s.record();
    mat_ops.inv(A.data, iA.data);
    inv_e.record();

    mat_ops.assign(A_save.data, A.data);
    timer_t det_s, det_e;
    det_s.record();
    auto detA = mat_ops.det(A.data);
    det_e.record();


    mat_ops.assign(A_save.data, A.data);
    std::cout << "matrix operations elapsed time = " << matrix_event_2.elapsed_time(matrix_event_1)/1000.0 << " s." << std::endl;

    std::cout << "gesv time = " << solve_e.elapsed_time(solve_s)/1000.0 << " s." << std::endl;
    std::cout << "inv time = " << inv_e.elapsed_time(inv_s)/1000.0 << " s." << std::endl;
    std::cout << "det time = " << det_e.elapsed_time(det_s)/1000.0 << " s." << std::endl;

    std::cout << "detA = " << detA << std::endl;
    std::cout << " norm(Ar) = " << vec_ops.norm(y) << std::endl;
    std::cout << " norm(A'r) = " << vec_ops.norm(z) << std::endl;

    mat_files.write_matrix("A_b.dat", A.data, 200);
    mat_files.write_matrix("B_b.dat", B.data, 200);
    mat_files.write_matrix("C_b.dat", C.data, 200);
    mat_files.write_matrix("D_b.dat", D.data, 200);
    mat_files.write_matrix("DTN_b.dat", DTN.data, 200);
    mat_files.write_matrix("D1_b.dat", D1.data, 200);
    mat_files.write_matrix("iA_b.dat", iA.data, 200);

    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << y[j] << " ";
    } 
    std::cout << std::endl;

    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << z[j] << " ";
    } 
    std::cout << std::endl;

    std::cout << " A(1:3)'x = ";
    for(size_t k=0;k<3;k++)
    {
        std::cout << r[k] << " ";
    }
    std::cout << std::endl;

    if(all_files_read)
    {
        mat_ops.geam('N', N, N, static_cast<T>(1.0), C_geam.data, static_cast<T>(-1.0), C.data, C.data);                  
        std::cout << "geam diff = " << mat_ops.norm_fro(C.data) << std::endl;
        mat_ops.geam('N', N, N, static_cast<T>(1.0), C_gemm.data, static_cast<T>(-1.0), D.data, D.data);                  
        std::cout << "gemm diff = " << mat_ops.norm_fro(D.data) << std::endl;
        mat_ops.geam('N', N, N, static_cast<T>(1.0), C1_gemm.data, static_cast<T>(-1.0), D1.data, D1.data);                  
        std::cout << "gemm1 diff = " << mat_ops.norm_fro(D1.data) << std::endl; 
        mat_ops.geam('N', N, N, static_cast<T>(1.0), CTN_gemm.data, static_cast<T>(-1.0), DTN.data, DTN.data);                  
        std::cout << "gemmTN diff = " << mat_ops.norm_fro(DTN.data) << std::endl;                
        mat_ops.geam('N', N, N, static_cast<T>(1.0), CNT_gemm.data, static_cast<T>(-1.0), DNT.data, DNT.data);                  
        std::cout << "gemmNT diff = " << mat_ops.norm_fro(DNT.data) << std::endl;
        vec_ops.assign(x, y);
        vec_ops.add_mul(static_cast<T>(-1.0), sol, y);
        std::cout << "||x-x_ref|| = " << vec_ops.norm(y) << std::endl;
        mat_ops.geam('N', N, N, static_cast<T>(1.0), iA_ref.data, static_cast<T>(-1.0), iA.data, iA.data);                  
        std::cout << "inv(A) diff = " << mat_ops.norm_fro(iA.data) << std::endl;
    }
    mat_ops.assign(rhs, y);
    mat_ops.gemv('N', A.data, static_cast<T>(-1.0), x, static_cast<T>(1.0), y);
    std::cout << "||Ax-b|| = " << vec_ops.norm(y) << std::endl;


    mat_ops.stop_use_matrices(P.data, A_save.data, A.data, B.data, C.data, D.data, D1.data, DTN.data, DNT.data, C_geam.data, C_gemm.data, CTN_gemm.data, C1_gemm.data, CNT_gemm.data, iA_ref.data, iA.data); 
    mat_ops.free_matrices(P.data, A_save.data, A.data, B.data, C.data, D.data, D1.data, DTN.data, DNT.data, C_geam.data, C_gemm.data, CTN_gemm.data, C1_gemm.data, CNT_gemm.data, iA_ref.data, iA.data);
    vec_ops.stop_use_vectors(x,y,z,r,rhs,sol); vec_ops.free_vectors(x,y,z,r,rhs,sol);

    return 0;
}