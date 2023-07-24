#ifndef __STABILITY_IRAM_PROCESS_HPP__
#define __STABILITY_IRAM_PROCESS_HPP__ 


#include <stability/IRAM/iram_container.h>
// #include <stability/IRAM/shift_bulge_chase.h>
#include <stability/IRAM/schur_select.h>
#include <iomanip>
#include <complex>
#include <stability/Galerkin_projection.h>

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
    using eigs_t = std::vector<std::complex<T> >;    
private:    
    using C = std::complex<T>;
    using container_t = stability::IRAM::iram_container<VectorOperations, MatrixOperations, Log>;
    // using bulge_t = stability::IRAM::shift_bulge_chase<VectorOperations, MatrixOperations, LapackOperations, Log>;
    using schur_select_t = stability::IRAM::schur_select<VectorOperations, MatrixOperations, LapackOperations, Log>;
    using project_t = stability::Galerkin_projection<VectorOperations, MatrixOperations, LapackOperations, LinearOperator, Log, eigs_t>;

public:
/**
 * @brief      Constructor.
 */
    iram_process(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, LapackOperations* lapack_, ArnoldiProcess* arnoldi_, SystemOperator* sys_op_, LinearOperator* A_, Log* log_):
    vec_ops_l(vec_ops_l_), mat_ops_l(mat_ops_l_), vec_ops_s(vec_ops_s_), mat_ops_s(mat_ops_s_), lapack(lapack_), sys_op(sys_op_), arnoldi(arnoldi_), A(A_), log(log_), f_set(false)
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
        // bulge = new bulge_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, log, lapack);
        schur_select = new schur_select_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, log, lapack);
        schur_select->set_target(which_sys_op);

        project = new project_t(vec_ops_l, mat_ops_l, vec_ops_s, mat_ops_s, lapack, A, log);


    }
    ~iram_process()
    {
        free_chk(project);
        free_chk(schur_select);
        free_chk(container);        
    }
    
    void set_target_eigs(const std::string& which_)
    {
        which = which_;
        project->set_target_eigs({which_sys_op, which} );
    }
    void set_number_of_desired_eigenvalues(unsigned int k0_)
    {
        k0 = k0_;
        schur_select->set_number_of_desired_eigenvalues(k0);

    }
    void set_linear_operator_stable_eigenvalues_halfplane(const T sign_) // sign_ = -1  => left half plane
    {
        sign = -sign_;
    } 
    void set_tolerance(const T tolerance_)
    {
        tolerance = tolerance_;//*vec_ops_l->get_l2_size();
        container->set_tolerance(tolerance);
    }
    void set_max_iterations(unsigned int max_it_)
    {
        max_iterations = max_it_;
    }
    void set_verbocity(bool verb_)
    {
        verbocity = verb_;
    }
    void set_initial_vector(const T_vec& x_0)
    {
        container->set_f(x_0);
        f_set = true;
    }
    
    eigs_t execute(T_vec v_init = nullptr)
    {
        std::vector<T_vec> fake_vec;
        return execute(fake_vec, fake_vec, v_init);
    }

    eigs_t execute(std::vector<T_vec>& eigvec_real, std::vector<T_vec>& eigvec_imag, T_vec v_init = nullptr)
    {
        T ritz_norm = T(1.0);
        unsigned int iters = 0;
        size_t k = 0;
        start_process();
        if(v_init == nullptr)
        {
            v_init = container->ref_f();
        }        
        while( (ritz_norm>tolerance)&&(iters<max_iterations) )
        {
            container->reset_ritz();
            // arnoldi->execute_arnoldi(k, container->ref_V(), container->ref_H(), v_init ); //(PA) V_j = V_j H_j + beta_j*f_{j+1}*e^T
            T ritz_value = arnoldi->execute_arnoldi_schur(k, container->ref_V(), container->ref_H(), v_init );
            // bulge->execute(*container);
            if(std::isnan(ritz_value))
            {
                throw std::runtime_error("iram_process: NAN value of the ritz estimate value is detected at step " + std::to_string(iters));
            }
            schur_select->execute(*container, ritz_value);
            container->to_gpu();
            ritz_norm = container->ritz_norm();
            k = container->K;
            if(verbocity)
            {
                log->info_f("iram_process: iteration = %i, k = %i, Ritz norm = %le", iters, k, ritz_norm);
                auto ritz_norms = container->get_ritz_norms();

                for(int j=0;j<k;j++)
                {
                    log->info_f("iram_process: ritz norm %i = %le", j, ritz_norms[j]);
                }
            }
            iters++;
        }

        // auto eigs = project->eigs(container->ref_V(), container->ref_H(), container->K0, eigvec_real, eigvec_imag ); 

        auto eigs = project->eigs(container->ref_V(), container->ref_H(), container->K0); 

        return eigs;
    }



private:
    bool verbocity = false;
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

    T_mat Q_fin_dev = nullptr;
    T_mat R_fin_dev = nullptr;
    unsigned int k0 = 6; // default value
    unsigned int m;
    size_t N;
    // bulge_t* bulge = nullptr;
    schur_select_t* schur_select = nullptr;
    container_t* container = nullptr;
    project_t* project = nullptr;
    T sign = T(1.0);
    T tolerance = T(1.0e-6);
    unsigned int max_iterations = 100; // default value
    bool f_set;
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
        if(!f_set)
        {
            container->init_f();
        }
    }

    void sort_eigs(eigs_t& eigs)
    {
        if((which == "LM")||(which == "lm"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const std::complex<T>& left_, const std::complex<T>& right_)
            {
             
                return std::abs(left_) > std::abs(right_);
            } 
            );
        }
        else if((which == "LR")||(which == "lr"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const std::complex<T>& left_, const std::complex<T>& right_)
            {
                return  left_.real() > right_.real();
            } 
            );
        }
        else if((which == "SR")||(which == "sr"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const std::complex<T>& left_, const std::complex<T>& right_)
            {
                return  left_.real() < right_.real();
            } 
            );
        }        
    }

        // mat_ops_l->make_zero_columns(container->ref_V(), k, m, container->ref_V() );
        // mat_ops_s->make_zero_columns(container->ref_H(), k, m, container->ref_H() );
        // std::vector<T> Q_fin(m*m, 0);
        // std::vector<T> U_fin(m*m, 0);
        // lapack->hessinberg_schur_from_gpu(container->ref_H(), m, Q_fin.data(), U_fin.data() );
        
        // host_2_device_cpy(Q_fin_dev, Q_fin.data(), m*m);
        
        // mat_ops_l->mat2column_mult_mat(container->ref_V(), Q_fin_dev, m, 1.0, 0.0, V1);

        // for(int j=0;j<k;j++)
        // {
        //     A->apply(&V1[j*N], &container->ref_V()[j*N]);
        // }
        // mat_ops_l->mat_T_gemm_mat_N(V1, container->ref_V(), m, 1.0, 0.0, R_fin_dev);
        // std::vector<T> R_fin(m*m, 0);
        // std::vector<T> R_fin_sub(k*k, 0);
        // device_2_host_cpy(R_fin.data(), R_fin_dev, m*m);
        // for(int j=0;j<k;j++)
        // {
        //     for(int l=0;l<k;l++)
        //     {
        //         R_fin_sub[j+l*k] = R_fin[j+l*m];
        //     }
        // }
        // eigs_t eigs = eigs_t(k, 0);
        // lapack->eigs(R_fin_sub.data(), k, eigs.data() );
        // sort_eigs(eigs);
};
}
}


#endif
