#ifndef __PERIODIC_ORBIT_HYPERPLANE_H__
#define __PERIODIC_ORBIT_HYPERPLANE_H__


namespace periodic_orbit
{
template<class VectorOperations, class NonlinearOperator>
class hyperplane
{

public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    hyperplane(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, const T_vec& x0_p, const T lambda_p):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p)
    {
        vec_ops_->init_vectors(x0_,n_,dx_,n0_reduced_, x_helper_); vec_ops_->start_use_vectors(x0_,n_,dx_,n0_reduced_, x_helper_);
        vec_ops_->init_vector(n_reduced_); vec_ops_->start_use_vector(n_reduced_, vec_ops_->get_default_size()-1 );
        update(0, x0_p, lambda_p);
    }
    ~hyperplane()
    {
        vec_ops_->stop_use_vectors(x0_, n_, dx_, x_helper_); vec_ops_->free_vectors(x0_, n_, dx_, x_helper_);
        vec_ops_->stop_use_vector(n_reduced_); vec_ops_->free_vector(n_reduced_);
    }

    void update(const T current_time, const T_vec& x0_p, const T lambda_p)
    {
        T_vec abs_vec_;
        vec_ops_->assign(x0_p, x0_);
        lambda_ = lambda_p;
        vec_ops_->init_vector(abs_vec_); vec_ops_->start_use_vector(abs_vec_);
        vec_ops_->make_abs_copy(n_, abs_vec_);
        index_ = vec_ops_->argmax_element(abs_vec_);
        nonlin_op_->F(current_time, x0_, lambda_, n_);
        vec_ops_->normalize(n_);
        vec_ops_->stop_use_vector(abs_vec_); vec_ops_->free_vector(abs_vec_);
        reduce(n_, n_reduced_);
        vec_ops_->assign(n_, n0_reduced_);
        reduce(n0_reduced_);
    }

    // R^{n} -> R^{n-1}
    void project_to(const T current_time, const T_vec& current_point, const T_vec& x0_p, T_vec& x1_p)const 
    {
        // vn1_proj = vn1 - f.*section_1.n'*vn1./(section_1.n'*f);
        nonlin_op_->F(current_time, current_point, lambda_, x_helper_);
        T projection_f = vec_ops_->scalar_prod(n_, x_helper_);
        T projection_v = vec_ops_->scalar_prod(n_, x0_p);
        vec_ops_->add_mul(1.0, x0_p, -projection_v/projection_f, x_helper_);
        vec_ops_->assign_skip_slices(x_helper_, {{index_,index_+1}}, x1_p);
    }

    // R^{n-1} -> R^{n}
    void restore_from(const T_vec& x0_p, T_vec& x1_p)const
    {
        //xk_val = - n_reduced'*(vec)./section.n(section.index, 1);
        T projection = -vec_ops_->scalar_prod(n_reduced_, x0_p)/(vec_ops_->get_value_at_point(index_, n_) );
        vec_ops_->set_value_at_point(projection, index_, x1_p);
    }


    // R^{n} -> R^{n} with 0 component
    void project_to(const T current_time, const T_vec& current_point, T_vec& x1_p)const 
    {
        nonlin_op_->F(current_time, current_point, lambda_, x_helper_);
        T projection_f = vec_ops_->scalar_prod(n_, x_helper_);
        T projection_v = vec_ops_->scalar_prod(n_, x1_p);
        // vec_ops_->add_mul(-projection_f/projection_v, x1_p, 1.0, x1_p);
        vec_ops_->add_mul(-projection_v/projection_f, x_helper_, 1.0, x1_p);
        vec_ops_->set_value_at_point(0.0, index_, x1_p);
    }

    // R^{n} with zero component -> R^{n}
    void restore_from(T_vec& x1_p)const
    {
        //xk_val = - n_reduced'*(vec)./section.n(section.index, 1);
        T projection = -vec_ops_->scalar_prod(n0_reduced_, x1_p)/(vec_ops_->get_value_at_point(index_, n_) );
        vec_ops_->set_value_at_point(projection, index_, x1_p);
    }


    bool is_crossed_in_normal_direction(const T_vec& x_from, const T_vec& x_to)const
    {
        //( ((section_1.n'*(xn1 - section_1.value))*(section_1.n'*(xn - section_1.value)))<0.0 )&&( section_1.n'*(xn1-xn)>0.0 )

        vec_ops_->add_mul(-1.0, x_from, 1.0, x_to, 0.0, dx_);
        bool flag_1 = (vec_ops_->scalar_prod(n_, dx_)>0.0);
        vec_ops_->add_mul(1.0, x_from, -1.0, x0_, 0.0, dx_);
        T dx_from = vec_ops_->scalar_prod(n_, dx_);
        vec_ops_->add_mul(1.0, x_to, -1.0, x0_, 0.0, dx_);
        T dx_to = vec_ops_->scalar_prod(n_, dx_);
        bool flag_2 = dx_from*dx_to<0.0;
        return flag_1&flag_2;

    }

    // retrns std::pair<T,T>: (error, distance)
    std::pair<T, T> intersection(const T_vec& x_from, const T_vec& x_to)const
    {
        // error = section.n'*(xn1 - section.value);
        // dinum = norm(section.n'*(xn1 - xn));
        vec_ops_->add_mul(1.0, x_to, -1.0, x0_, 0.0, dx_);
        T error = vec_ops_->scalar_prod(n_, dx_);
        vec_ops_->add_mul(-1.0, x_from, 1.0, x_to, 0.0, dx_);
        T distance = std::abs( vec_ops_->scalar_prod(n_, dx_) );

        return {error, distance};
    }

    void get_initial_point(T_vec& x_p)const
    {
        vec_ops_->assign(x0_, x_p);
    }

private:
    T_vec x0_;
    mutable T_vec x_helper_;    
    T_vec n_;
    T_vec n0_reduced_;    //of size n, but with a zero in the "index_" index
    T_vec n_reduced_;
    mutable T_vec dx_;
    T lambda_;
    size_t index_;
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;

    void reduce(const T_vec& x_from, T_vec& x_to)
    {
        vec_ops_->assign_skip_slices(x_from, {{index_,index_+1}}, x_to);
    }

    void reduce(T_vec& x1_p)
    {
        vec_ops_->set_value_at_point(0.0, index_, x1_p);
    }


};


}





#endif