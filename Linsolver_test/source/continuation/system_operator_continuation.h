#ifndef __CONTINUATION__SYSTEM_OPERATOR_CONTINUATION_H__
#define __CONTINUATION__SYSTEM_OPERATOR_CONTINUATION_H__

#include <string>
#include <stdexcept>


namespace continuation
{
template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSystemSolver>
class system_operator_continuation
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    system_operator_continuation(VectorOperations* vec_ops_, LinearOperator* lin_op_, LinearSystemSolver* SM_solver_):
    vec_ops(vec_ops_),
    lin_op(lin_op_),
    SM_solver(SM_solver_)
    {
        vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
        vec_ops->init_vector(f); vec_ops->start_use_vector(f);
        vec_ops->init_vector(Jlambda); vec_ops->start_use_vector(Jlambda);
    }
 
    ~system_operator_continuation()
    {
        vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
        vec_ops->stop_use_vector(f); vec_ops->free_vector(f);
        vec_ops->stop_use_vector(Jlambda); vec_ops->free_vector(Jlambda);
    }

    void set_tangent_space(T_vec& x_0_, T& lambda_0_, T_vec& x_0_s_, T& lambda_0_s_, T& ds_l_)
    {
        x_0 = x_0_;
        lambda_0 = lambda_0_;
        x_0_s = x_0_s_;
        lambda_0_s = lambda_0_s_;
        tangent_set = true;
        ds_l = ds_l_;
    }
    
    bool update_tangent_space(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& x_1_s, T& lambda_1_s)
    {
        
        bool flag_lin_solver = false;
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->jacobian_alpha(Jlambda);
        vec_ops->assign_scalar(T(0), f);
        T beta = T(1);
        T alpha = lambda_0_s;
        SM_solver->get_linsolver_handle()->monitor().set_temp_tolerance(T(1.0e-8));
        flag_lin_solver = SM_solver->solve((*lin_op), x_0_s, Jlambda, alpha, f, beta, x_1_s, lambda_1_s);
        SM_solver->get_linsolver_handle()->monitor().restore_tolerance();
        T norm = vec_ops->norm_rank1(x_1_s, lambda_1_s);
        lambda_1_s/=norm;
        vec_ops->scale(norm, x_1_s);
        return flag_lin_solver;
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x, T& d_lambda)
    {

        bool flag_lin_solver = false;
        if(tangent_set)
        {            
            /*
                Flambda = @(x,param)PP.operator_lambda(x,param);
                f_lambda=Flambda(x1, lambda1);
                
                orthogonal_projection = (x1-x0)'*x0_s+(lambda1-lambda0)*lambda0_s - delta_s;
                
                b = -PP.F(x1, lambda1);
                beta = -orthogonal_projection;
                d = f_lambda;
                c = x0_s;
                alpha = lambda0_s;

            */
            nonlin_op->set_linearization_point(x, lambda);
            nonlin_op->jacobian_alpha(Jlambda);
            nonlin_op->F(x, lambda, f);            
            
            vec_ops->add_mul_scalar(T(0), T(-1), f); //f=-F(x,lambda)
            T beta =  - orthogonal_projection(x, lambda); //beta = -orth_proj
            T alpha = lambda_0_s;

            flag_lin_solver = SM_solver->solve((*lin_op), x_0_s, Jlambda, alpha, f, beta, d_x, d_lambda);
            
            
        }  
        else
        {
            throw std::runtime_error(std::string("Continuation::system_operator " __FILE__ " " __STR(__LINE__) " tangent space is not set. Set it with the method set_tangent_space(...).") );
        }      
        return flag_lin_solver;

    }


private:
    VectorOperations* vec_ops;
    LinearOperator* lin_op;
    LinearSystemSolver* SM_solver;
    bool tangent_set = false;
    T_vec x_0, x_0_s;
    T lambda_0, lambda_0_s;
    T_vec dx, f, Jlambda;
    T ds_l;
    
    //orthogonal_projection = (x1-x0)'*x0_s+(lambda1-lambda0)*lambda0_s - delta_s;
    T orthogonal_projection(const T_vec& x_1, const T& lambda_1)
    {
/*
    
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z)const;
*/      size_t sz =  vec_ops->get_vector_size();
        vec_ops->assign_mul(T(1), x_1, T(-1), x_0, dx); //dx = x_1-x_0
        T x_proj = vec_ops->scalar_prod(dx, x_0_s); //(dx,x_0_s)
        T lambda_proj = (lambda_1 - lambda_0)*lambda_0_s;
        return (x_proj + lambda_proj - ds_l);

    }



};



}

#endif