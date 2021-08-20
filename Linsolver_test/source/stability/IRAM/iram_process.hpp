#ifndef __STABILITY_IRAM_PROCESS_HPP__
#define __STABILITY_IRAM_PROCESS_HPP__ 


#include <stability/IRAM/iram_container.h>
#include <stability/IRAM/shift_bulge_chase.h>
#include <iomanip>

/**
 * @brief      implementaton of the IRAM
 *
 * @tparam     VectorOperations  { vector operations class }
 * @tparam     MatrixOperations  { matrix operations class }
 * @tparam     LapackOperations  { lapack wrap }
 * @tparam     ArnoldiProcess    { arnolid method that generates V,H and f. The preconditioned LinearOperator is executed from the ArnoldiProcess. }
 * @tparam     SystemOperator    { A linear operator that is executed in the ArnoldiProcess and contains the target for the sought eigenvalues (e.g. we can set LR by set_target_eigs(), but for the shift-inverse one needs LM. These LM are returned by the SystemOperator class by the method  std::string target_eigs() ) }
 * @tparam     LinearOperator    { Original(!!!) linear operator A that assumes the "apply" method }
 * @tparam     Log               { Log class }
 */
namespace stability
{
namespace IRAM
{

template<class VectorOperations, class MatrixOperations, class LapackOperations, class ArnoldiProcess, class SystemOperator, class LinearOperator, class Log>
class iram_process
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
public:
    using eigs_t = std::vector<std::pair<T,T> >;    
private:    
    using C = std::complex<T>;
    using container_t = stability::IRAM::iram_container<VectorOperations, MatrixOperations, Log>;
    using bulge_t = stability::IRAM::shift_bulge_chase<VectorOperations, MatrixOperations, LapackOperations, Log>;

public:
/**
 * @brief      Constructor.
 */
    iram_process(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, LapackOperations* lapack_, ArnoldiProcess* arnoldi_, SystemOperator* sys_op_, LinearOperator* A_, Log* log_):
    vec_ops_l(vec_ops_l_), mat_ops_l(mat_ops_l_), vec_ops_s(vec_ops_s_), mat_ops_s(mat_ops_s_), lapack(lapack_), sys_op(sys_op_), arnoldi(arnoldi_), A(A_), log(log_)
    {
        auto small_rows = mat_ops_s->get_rows();
        auto small_cols = mat_ops_s->get_cols();
        auto large_rows = mat_ops_l->get_rows();
        auto large_cols = mat_ops_l->get_cols();
        if(small_cols != small_rows)
        {
            throw std::logic_error("small matrix is not square");
        }
        if(small_rows != large_cols)
        {
            throw std::logic_error("large_cols=" + std::to_string(large_cols) + " != small_rows=" + std::to_string(small_rows) );
        }
        N = large_rows;
        m = small_rows;

        which_sys_op = sys_op->target_eigs();
        container = new container_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, log);
        bulge = new bulge_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, log, lapack);
        bulge->set_target(which_sys_op);
        mat_ops_l->init_matrix(V1); mat_ops_l->start_use_matrix(V1);
        mat_ops_s->init_matrix(Q_fin_dev); mat_ops_s->start_use_matrix(Q_fin_dev);
        mat_ops_s->init_matrix(R_fin_dev); mat_ops_s->start_use_matrix(R_fin_dev);
        
    }
    ~iram_process()
    {
        free_chk(bulge);
        free_chk(container);
        if(V1 != nullptr)
        {
            mat_ops_l->stop_use_matrix(V1); mat_ops_l->free_matrix(V1);
        }
        if(Q_fin_dev != nullptr)
        {
            mat_ops_s->stop_use_matrix(Q_fin_dev); mat_ops_s->free_matrix(Q_fin_dev);
        }
        if(R_fin_dev != nullptr)
        {
            mat_ops_s->stop_use_matrix(R_fin_dev); mat_ops_s->free_matrix(R_fin_dev);
        }        
    }
    
    void set_target_eigs(const std::string& which_)
    {
        which = which_;
    }
    void set_number_of_desired_eigenvalues(unsigned int k0_)
    {
        k0 = k0_;
        bulge->set_number_of_desired_eigenvalues(k0);
    }
    void set_linear_operator_stable_eigenvalues_halfplane(const T sign_) // sign_ = -1  => left half plane
    {
        sign = -sign_;
    } 
    void set_tolerance(const T tolerance_)
    {
        tolerance = tolerance_;//*vec_ops_l->get_l2_size();
    }
    void set_max_iterations(unsigned int max_it_)
    {
        max_iterations = max_it_;
    }

    eigs_t execute()
    {
        T ritz_norm = T(1.0);
        unsigned int iters = 0;
        size_t k = 0;
        start_process();
        while( (ritz_norm>tolerance)&&(iters<max_iterations) )
        {
            container->reset_ritz();
            arnoldi->execute_arnoldi(k, container->ref_V(), container->ref_H(), container->ref_f() ); //(PA) V_j = V_j H_j + beta_j*f_{j+1}*e^T
            bulge->execute(*container);
            container->to_gpu();
            ritz_norm = container->ritz_norm();
            k = container->K;
            log->info_f("iram_process: teration = %i, k = %i, Ritz norm = %le", iters, k, ritz_norm);
            iters++;
        }
        
        // obtaining the original eigenvlaues
        mat_ops_l->make_zero_columns(container->ref_V(), k, m, container->ref_V() );
        mat_ops_l->make_zero_columns(V1, k, m, V1 );
        mat_ops_s->make_zero_columns(container->ref_H(), k, m, container->ref_H() );
        std::vector<T> Q_fin(m*m, 0);
        std::vector<T> U_fin(m*m, 0);
        lapack->hessinberg_schur_from_gpu(container->ref_H(), m, Q_fin.data(), U_fin.data() );
        
        host_2_device_cpy(Q_fin_dev, Q_fin.data(), m*m);
        
        mat_ops_l->mat2column_mult_mat(container->ref_V(), Q_fin_dev, m, 1.0, 0.0, V1);

        for(int j=0;j<k;j++)
        {
            A->apply(&V1[j*N], &container->ref_V()[j*N]);
        }
        mat_ops_l->mat_T_gemm_mat_N(V1, container->ref_V(), m, 1.0, 0.0, R_fin_dev);
        std::vector<T> R_fin(m*m, 0);
        std::vector<T> R_fin_sub(k*k, 0);
        device_2_host_cpy(R_fin.data(), R_fin_dev, m*m);
        for(int j=0;j<k;j++)
        {
            for(int l=0;l<k;l++)
            {
                // std::cout << R_fin[j+l*m] << " ";
                R_fin_sub[j+l*k] = R_fin[j+l*m];
            }
            // std::cout << std::endl;
        }
        std::vector<T> eig_real(k*k, 0);
        std::vector<T> eig_imag(k*k, 0);
        lapack->eigs(R_fin_sub.data(), k, eig_real.data(), eig_imag.data() );
        
        for(int j=0;j<k;j++)
        {
            eigs.push_back({eig_real[j], eig_imag[j]});
        }
        sort_eigs();
        
        std::cout << std::scientific;
        for(int j=0;j<k;j++)
        {
            std::cout << eigs[j].first << " " << eigs[j].second << std::endl;
        }

        return eigs;
    }



private:
    std::string which = "LR"; // default
    std::string which_sys_op = "LM"; // default
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    LapackOperations* lapack;
    ArnoldiProcess* arnoldi;
    SystemOperator* sys_op;
    LinearOperator* A;
    Log* log;
    eigs_t eigs;

    T_mat V1 = nullptr;
    T_mat Q_fin_dev = nullptr;
    T_mat R_fin_dev = nullptr;
    unsigned int k0 = 6; // default value
    unsigned int m;
    size_t N;
    bulge_t* bulge = nullptr;
    container_t* container = nullptr;
    T sign = T(1.0);
    T tolerance = T(1.0e-6);
    unsigned int max_iterations = 100; // default value

    template<class T>
    void free_chk(T ref_)
    {
        if(ref_ != nullptr)
        {
            delete ref_;
        }
    }

    void start_process()
    {
        container->force_gpu();
        container->set_f();
    }

    void sort_eigs()
    {
        if((which == "LM")||(which == "lm"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const std::pair<T,T>& left_, const std::pair<T,T>& right_)
            {
                auto left2_ = left_.first*left_.first+left_.second*left_.second;
                auto right2_ = right_.first*right_.first+right_.second*right_.second;
                return left2_ > right2_;
            } 
            );
        }
        else if((which == "LR")||(which == "lr"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const std::pair<T,T>& left_, const std::pair<T,T>& right_)
            {
                return  left_.first > right_.first;
            } 
            );
        }
    }


};
}
}


#endif
