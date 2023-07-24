#ifndef __STABILITY_IRAM_IRAM_CONTAINER_H__
#define __STABILITY_IRAM_IRAM_CONTAINER_H__

#include <vector>
#include <common/gpu_matrix_file_operations.h>
#include <utils/cuda_support.h>


namespace stability
{
namespace IRAM
{

template<class VectorOperations, class MatrixOperations, class Log>
class iram_container
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;    

    VectorOperations* vec_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_l;
    MatrixOperations* mat_ops_s;
    Log* log;

public:
    iram_container(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, T tolerance_p = 1.0e-6, size_t K_p = 6, bool debug_ = false):
    vec_ops_l(vec_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_l(mat_ops_l_),
    mat_ops_s(mat_ops_s_),
    log(log_),
    tolerance(tolerance_p),
    K(K_p),
    K0(K_p),
    debug(debug_)
    {
        mat_ops_l->init_matrix(V_gpu); mat_ops_l->start_use_matrix(V_gpu);
        mat_ops_s->init_matrix(H_gpu); mat_ops_s->start_use_matrix(H_gpu);
        vec_ops_l->init_vector(f_gpu); vec_ops_l->start_use_vector(f_gpu);
        N = mat_ops_l->get_rows();
        m = mat_ops_l->get_cols();
        // std::cout << "iram_container: N = " << N << " m = " << m << std::endl;
        H_cpu = std::vector<T>(m*m,0);
        ritz = std::vector<T>(m,0);
        if(debug)
        {
            V_cpu = std::vector<T>(N*m, 0);
            f_cpu = std::vector<T>(N, 0);
        }

    }
    ~iram_container()
    {
        free_mat_l(V_gpu);
        free_mat_s(H_gpu);
        free_vec_l(f_gpu);
    }
    void set_tolerance(const T tolerance_)
    {
        tolerance = tolerance_;//*vec_ops_l->get_l2_size();
    }
    T get_tolerance()const
    {
        return tolerance;
    }    
    void set_f(const T_vec& x)
    {
        if(debug) log->info("iram_container: f-vector set from external vector");
        if(f_gpu != nullptr)
        {
            vec_ops_l->assign(x, f_gpu);
        }
    }
    void init_f(bool random_ = false)
    {
        if(debug) log->info("iram_container: initialized f-vector");
        vec_ops_l->assign_scalar(0, f_gpu);
        vec_ops_l->set_value_at_point(1, 0, f_gpu);
        if(random_)
        {
            vec_ops_l->assign_random(f_gpu);
        }
    }
    void to_gpu()
    {
        if(debug) log->info("iram_container: copy to gpu");
        host_2_device_cpy(H_gpu, H_cpu.data(), m*m);
        if(debug)
        {
            host_2_device_cpy(V_gpu, V_cpu.data(), N*m);
            host_2_device_cpy(f_gpu, f_cpu.data(), N);
        }
        cpu_active = false;
    }
    void to_cpu()
    {
        if(debug) log->info("iram_container: copy to cpu");
        device_2_host_cpy(H_cpu.data(), H_gpu, m*m);
        if(debug)
        {
            device_2_host_cpy(V_cpu.data(), V_gpu, N*m);
            device_2_host_cpy(f_cpu.data(), f_gpu, N);
        }
        cpu_active = true;
    }
    T_mat ref_V()
    {
        if(cpu_active&&debug)
        {
            return V_cpu.data();
        }
        else
        {
            return V_gpu;
        }
    }
    T_mat ref_f()
    {
        if(cpu_active&&debug)
        {
            return f_cpu.data();
        }
        else
        {
            return f_gpu;
        }
    }
    T_mat ref_H()
    {
        if(cpu_active)
        {
            return H_cpu.data();
        }
        else
        {
            return H_gpu;
        }
    }


    void force_gpu()
    {
        if(debug) log->info("iram_container: forcing pointers on gpu.");
        cpu_active = false;
    }
    void force_cpu()
    {
        if(debug) log->info("iram_container: forcing pointers on cpu.");
        cpu_active = true;
    }

    void reset_ritz()
    {
        if(debug) log->info("iram_container: ritz vector norms reset.");
        for(int j=0;j<m;j++) ritz.at(j) = T(0);
    }
    std::vector<T> get_ritz_norms()
    {
        return ritz;
    }

    T ritz_norm()
    {
        T norm = 0;
        for(int j=0;j<K0;j++) norm+=ritz.at(j)*ritz.at(j);

        return std::sqrt(norm);
    }

    std::vector<T> ritz;
    size_t K;
    size_t K0;

private:
    bool cpu_active = false;
    bool debug;
    size_t m;
    size_t N;
    T_mat V_gpu = nullptr;
    std::vector<T> V_cpu;
    T_mat H_gpu = nullptr;
    std::vector<T> H_cpu;
    T_vec f_gpu = nullptr;
    std::vector<T> f_cpu;
    T tolerance = 1.0e-6;

    void free_mat_l(T_mat mat_)
    {
        if(mat_ != nullptr)
        {
            mat_ops_l->stop_use_matrix(mat_); mat_ops_l->free_matrix(mat_);
            mat_ = nullptr;
        }
    }
    void free_mat_s(T_mat mat_)
    {
        if(mat_ != nullptr)
        {
            mat_ops_s->stop_use_matrix(mat_); mat_ops_s->free_matrix(mat_);
            mat_ = nullptr;
        }        
    }
    void free_vec_l(T_vec vec_)
    {
        if(vec_ != nullptr)
        {
            vec_ops_l->stop_use_vector(vec_); vec_ops_l->free_vector(vec_);
            vec_ = nullptr;
        }         
    }
    void free_vec_s(T_vec vec_)
    {
        if(vec_ != nullptr)
        {
            vec_ops_s->stop_use_vector(vec_); vec_ops_s->free_vector(vec_);
            vec_ = nullptr;
        }         
    }    


};

}
}


#endif