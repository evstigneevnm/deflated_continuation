#include <boost/multiprecision/cpp_bin_float.hpp>
#include <iostream>
#include <common/cpu_vector_operations_var_prec.h>
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
    

    using vec_ops_t = cpu_vector_operations_var_prec<200>;
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
    vec_t x, y, z, r;
    vec_ops.init_vectors(x,y,z,r); vec_ops.start_use_vectors(x,y,z,r);
    vec_ops.assign_scalar(1, x);
    vec_ops.assign_scalar(boost::multiprecision::sqrt(static_cast<T>(2.0)), y);

    std::cout << " norm(x) = " << vec_ops.norm(x) << std::endl;
    std::cout << " norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.normalize(y);
    std::cout << " normalized y norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.add_mul_scalar(2.0, boost::multiprecision::sqrt(static_cast<T>(2.0)), y);
    std::cout << " add mul scalar y norm(y) = " << vec_ops.norm(y) << std::endl;
    vec_ops.assign(y, z);
    std::cout << " assigned y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    std::cout << " is result valid?: " << vec_ops.check_is_valid_number(z) << std::endl;
    vec_ops.assign_mul(3.0, y, z);
    std::cout << " assign_mul 3y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    vec_ops.assign_mul(6.0, x, -3.0, y, z);
    std::cout << " assign_mul 6x-3y->z norm(z) = " << vec_ops.norm(z) << std::endl;
    vec_ops.set_value_at_point(100.0, 1, z);
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
    vec_ops.assign_random(r,-100.0,100.0);
    std::cout << " random r norm(r) = " << vec_ops.norm(r) << std::endl;
    for(size_t j = 0; j < 10; ++j)
    {
        std::cout << r[j] << " ";
    }
    std::cout << std::endl;

    mat_ops_t mat_ops(N, N, &vec_ops);
    mat_files_t mat_files(&mat_ops);

    matrix_t A(N,N), B(N,N), C(N,N), C_geam(N,N);

    mat_ops.init_matrices(A.data, B.data, C.data, C_geam.data); 
    mat_ops.start_use_matrices(A.data, B.data, C.data, C_geam.data);

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
        all_files_read = true;
    }
    catch(const std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Using random data." << std::endl;
        mat_ops.assign_random(A.data);
        mat_ops.assign_random(B.data);        
    }

    timer_t matrix_event_1, matrix_event_2;
    matrix_event_1.record();



    mat_ops.gemv('N', A.data, 1.0, x, 0.0, y);
    mat_ops.gemv('T', A.data, 1.0, x, 0.0, z);
    mat_ops.mat2column_dot_vec(A.data, 3, 1.0, x, 0.0, r);

    mat_ops.geam('N', N, N, 1.0, A.data, 1.0, B.data, C.data);


    matrix_event_2.record();
    std::cout << "matrix operations elapsed time = " << matrix_event_2.elapsed_time(matrix_event_1)/1000.0 << " s." << std::endl;
    std::cout << " norm(Ar) = " << vec_ops.norm(y) << std::endl;
    std::cout << " norm(A'r) = " << vec_ops.norm(z) << std::endl;

    mat_files.write_matrix("A_b.dat", A.data, 200);
    mat_files.write_matrix("B_b.dat", B.data, 200);
    mat_files.write_matrix("C_b.dat", C.data, 200);

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
        mat_ops.geam('N', N, N, 1.0, C_geam.data, -1.0, C.data, C.data);                  
        std::cout << "geam diff = " << mat_ops.norm_fro(C.data) << std::endl;
    }
    
    


    mat_ops.stop_use_matrices(A.data, B.data, C.data, C_geam.data); mat_ops.free_matrices(A.data, B.data, C.data, C_geam.data);
    vec_ops.stop_use_vectors(x,y,z,r); vec_ops.free_vectors(x,y,z,r);

    return 0;
}