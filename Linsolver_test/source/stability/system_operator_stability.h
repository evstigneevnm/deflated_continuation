#ifndef __STABILITY__SYSTEM_OPERATOR_STABILITY_H__
#define __STABILITY__SYSTEM_OPERATOR_STABILITY_H__
    

/**
*
*   Class that implements particular funciton over the linear operator 
*   to accelerate eigenvalues solver.
*   
*   This particular class uses shift-inverse method with zero shift (i.e. looking at eigenvalues near (0,0)
*
*/


//for the estimaiton of the original eigenvalues
// #include <thrust/complex.h>

namespace stability
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSystemSolver, class Log>
class system_operator_stability
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    system_operator_stability(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_, LinearOperator* lin_op_, LinearSystemSolver* lin_solv_, Log* log_):
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_),
    lin_op(lin_op_),
    lin_solv(lin_solv_),
    log(log_)
    {
        tolerance = 1.0e-5;
        max_iterations = 5000;
        vec_ops->init_vector(r_v); vec_ops->start_use_vector(r_v);
        vec_ops->init_vector(dx_v); vec_ops->start_use_vector(dx_v);
    }

    ~system_operator_stability()
    {
        vec_ops->stop_use_vector(r_v); vec_ops->free_vector(r_v);
        vec_ops->stop_use_vector(dx_v); vec_ops->free_vector(dx_v);
    }
    void set_tolerance(T tolerance_)
    {
        tolerance = tolerance_;
    }
    void set_max_iterations(unsigned int max_iter_)
    {
        max_iterations = max_iter_;
    }

    bool solve(const T_vec& v_in, T_vec& v_out)
    {
        if(!linearization_set) throw std::runtime_error("system_operator_stability: linearization point not set");

        
        T tolerance_local = tolerance*vec_ops->get_l2_size();
        lin_solv->monitor().set_temp_max_iterations(max_iterations);

        bool linear_system_converged = false;
        int iter = 0;
        T factor = T(1.0);
        while((!linear_system_converged)&&(iter<max_retries))
        {
            vec_ops->assign_scalar(T(0.0), v_out);  //THIS IS A MUST for an inexact arnold process
            lin_solv->monitor().set_temp_tolerance(tolerance_local*factor);

            linear_system_converged = lin_solv->solve((*lin_op), v_in, v_out);   

            T minimum_resid = lin_solv->monitor().resid_norm_out();
            int iters_performed = lin_solv->monitor().iters_performed();
            
            lin_solv->monitor().restore_max_iterations();
            lin_solv->monitor().restore_tolerance(); 

            if(!linear_system_converged)
                log->warning_f("system_operator_stability: linear solver failed, desired residual = %le, minimum attained residual = %le with %i iterations. Retrying...", tolerance_local*factor, minimum_resid, iters_performed); 
            iter++;
            factor*=T(10.0);  
        }
        if(!linear_system_converged)
        {
            throw std::runtime_error("system_operator_stability: linear solver failed to converge."); 
        }
        
    
      

        return linear_system_converged;
    }

    void set_linerization_point(const T_vec& v_0, const T lambda)
    {
        nonlin_op->set_linearization_point(v_0, lambda);
        linearization_set = true;
    }

    // // eigenvalues are on the host computer.
    // void map_eigenvalues_to_original(std::vector< thrust::complex<T> >& eigenvalues)
    // {
    //     for(auto &x: eigenvalues)
    //     {
            
    //     }

    // }

private:
    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    LinearOperator* lin_op;
    LinearSystemSolver* lin_solv;
    Log* log;

    T tolerance;
    unsigned int max_iterations;
    unsigned int max_retries = 4;
    bool linearization_set = false;
    T_vec r_v;
    T_vec dx_v;


    /*
    //This approach doesn't work any better than the plain system solution

    // call to this funciton: linear_system_converged = iterate_lin_solver(v_in, v_out);


    bool iterate_lin_solver(const T_vec& u_in, T_vec& u_out)
    {
        T rhs_norm = 1.0;
        int iter = 0;
        bool iterations_converged = false;
        vec_ops->assign_scalar(T(0.0), dx_v);
        while((rhs_norm>1.0e-6)&&(iter<20))
        {
            form_rhs(u_in, u_out, r_v);
            lin_solv->solve((*lin_op), r_v, dx_v);
            vec_ops->add_mul(T(1.0), dx_v, T(1.0), u_out); 
            rhs_norm = vec_ops->norm(r_v);
            
            log->info_f("system_operator_stability.iterate_lin_solver: resid = %le, iter = %i", double(rhs_norm), iter);

            iter++;
            
        }

        if(rhs_norm <= 1.0e-6)
            iterations_converged = true;

        return iterations_converged;

    }

    void form_rhs(const T_vec& u_in, const T_vec& u_out, T_vec& r_out) const
    {
        lin_op->apply(u_out, r_out);
        //y := mul_x*x + mul_y*y
        vec_ops->add_mul(T(1.0), u_in, T(-1.0), r_out);
    }
    */
};

}

#endif