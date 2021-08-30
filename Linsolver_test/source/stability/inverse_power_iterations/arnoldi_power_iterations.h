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
#include <complex>
//debug
#include <common/gpu_matrix_file_operations.h>
#include <utils/cuda_support.h>
#include <stability/Galerkin_projection.h>

namespace stability
{


template<class VectorOperations, class MatrixOperations, class ArnoldiProcess, class LapackOperations, class LinearOperator, class Log>
class arnoldi_power_iterations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
    typedef gpu_matrix_file_operations<MatrixOperations> files_mat_t;
    
    typedef std::vector< std::complex<T> > eigs_t;

    using project_t = stability::Galerkin_projection<VectorOperations, MatrixOperations, LapackOperations, LinearOperator, Log, eigs_t>;

    arnoldi_power_iterations(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, ArnoldiProcess* arnolid_proc_, LapackOperations* lapack_, LinearOperator* A_):
    vec_ops_l(vec_ops_l_),
    mat_ops_l(mat_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_s(mat_ops_s_),    
    log(log_),
    arnolid_proc(arnolid_proc_),
    lapack(lapack_),
    A(A_)
    {
        small_rows = mat_ops_s->get_rows();
        small_cols = mat_ops_s->get_cols();
        large_rows = mat_ops_l->get_rows();
        large_cols = mat_ops_l->get_cols();

        project = new project_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, lapack, A, log);
        start_all();


    }
    


    ~arnoldi_power_iterations()
    {

        if(project != nullptr)
        {
            delete project;
        }
        stop_all();    
    }

    void set_linear_operator_stable_eigenvalues_halfplane(const T sign_) // sign_ = -1  => left half plane
    {
        sign = -sign_;
    }  

    eigs_t execute(std::string which = "LR")
    {
        size_t k = 0;
        arnolid_proc->execute_arnoldi(k, V_mat, H_mat);
        project->set_target_eigs(which);
        // eigs_t eigs(small_cols, 0);
        auto eigs = project->eigs(V_mat, H_mat, small_cols);
        // lapack->hessinberg_eigs_from_gpu(H_mat, small_rows, eigs.data() );
        // eigs_t eigs;
        // std::vector<T> eigs_magnitude;
        // eigs.reserve(small_rows);

        // T max_re_eig = 0.0; 
        // for(int j=0;j<small_rows;j++)
        // {
        //     T eig_real_l = sign*eig_real[j];
        //     T eig_imag_l = eig_imag[j];
        //     eigs.push_back( {eig_real_l, eig_imag_l} );
        //     eigs_magnitude.push_back(eig_real_l*eig_real_l+eig_imag_l*eig_imag_l);

        //     if(max_re_eig<eig_real_l)
        //         max_re_eig = eig_real_l;
        // }
        
        // std::sort(eigs.rbegin(), eigs.rend() ); //from max to min by the first pair


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
    project_t* project = nullptr;
    LinearOperator* A;

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