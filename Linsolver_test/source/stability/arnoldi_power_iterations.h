#ifndef __STABILITY__ARNOLDI_POWER_ITERATIONS_H__
#define __STABILITY__ARNOLDI_POWER_ITERATIONS_H__


/**
*
*   The method implements Arnoldi power iterations. 
*   The linarizaiton point is set in the main class 'stability' by triggering it in the system_perator_stability class
*   before the call to the execute method.
*
*/

#include <vector>
#include <algorithm>

//debug
#include <common/gpu_matrix_file_operations.h>
#include <utils/cuda_support.h>

namespace stability
{


template<class VectorOperations, class MatrixOperations, class ArnoldiProcess, class LapackOperations, class Log>
class arnoldi_power_iterations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
    typedef gpu_matrix_file_operations<MatrixOperations> files_mat_t;
    
    typedef std::vector<std::pair<T,T>> eigs_t;


    arnoldi_power_iterations(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, ArnoldiProcess* arnolid_proc_, LapackOperations* lapack_):
    vec_ops_l(vec_ops_l_),
    mat_ops_l(mat_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_s(mat_ops_s_),    
    log(log_),
    arnolid_proc(arnolid_proc_),
    lapack(lapack_)
    {
        small_rows = mat_ops_s->get_rows();
        small_cols = mat_ops_s->get_cols();
        large_rows = mat_ops_l->get_rows();
        large_cols = mat_ops_l->get_cols();

        start_all();
        

    }
    


    ~arnoldi_power_iterations()
    {

        stop_all();
    
    }

    void set_liner_operator_stable_eigenvalues_halfplane(const T sign_)
    {
        sign = -sign_;
    }  

    eigs_t execute()
    {
        arnolid_proc->execute_arnoldi(0, V_mat, H_mat);
        lapack->hessinberg_eigs_from_gpu(H_mat, small_rows, eig_real, eig_imag);
        eigs_t eigs;
        eigs.reserve(small_rows);
        for(int j=0;j<small_rows;j++)
        {
            eigs.push_back(std::make_pair(sign*eig_real[j], eig_imag[j]));
        }
        
        std::sort(eigs.rbegin(), eigs.rend()); //from max to min by the first pair

//        TODO: temporal checking 
/*
        for(auto &x: eigs)
        {
            T re = x.first;
            T im = x.second;
            if(im>=0.0)
                std::cout << re << " +" << im << " i" << std::endl;
            else
                std::cout << re << " " << im << " i" << std::endl;
        }
*/

/*        
        T_mat vec_cpu = host_allocate<T>(large_rows*large_cols);
        device_2_host_cpy<T>(vec_cpu, V_mat, large_rows*large_cols);


        for(int j=0;j<large_rows;j++)
        {
            for(int k=0;k<large_cols;k++)
            {
                std::cout << vec_cpu[I2_R(j,k,large_rows)] << " ";
            }
            std::cout << std::endl;
        }

        free(vec_cpu);
*/        
        return eigs;
            
    }



private:
    files_mat_t *files_mat_small;

    Log* log;
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    ArnoldiProcess* arnolid_proc;
    LapackOperations* lapack;

    T_mat V_mat;
    T_mat H_mat;

    size_t small_rows;
    size_t small_cols;
    size_t large_rows;
    size_t large_cols;

    T* eig_real;
    T* eig_imag;

    T sign = T(1.0);

    void start_all()
    {
        mat_ops_l->init_matrix(V_mat); mat_ops_l->start_use_matrix(V_mat);
        mat_ops_s->init_matrix(H_mat); mat_ops_s->start_use_matrix(H_mat);
        eig_real = (T*) malloc(sizeof(T)*small_rows);
        eig_imag = (T*) malloc(sizeof(T)*small_rows);
    }
    void stop_all()
    {
        mat_ops_l->stop_use_matrix(V_mat); mat_ops_l->free_matrix(V_mat);
        mat_ops_s->stop_use_matrix(H_mat); mat_ops_s->free_matrix(H_mat);
        free(eig_real);
        free(eig_imag);
    }



};

}

#endif // __STABILITY__ARNOLDI_POWER_ITERATIONS_H__