#ifndef __PERIODIC_ORBIT_GLUED_NONLINEAR_OPERATOR_AND_JACOBIAN_H__
#define __PERIODIC_ORBIT_GLUED_NONLINEAR_OPERATOR_AND_JACOBIAN_H__

#include <memory>
#include <vector>
#include <common/glued_vector_operations.h>
#include <common/glued_vector_space.h>

namespace periodic_orbit
{
namespace detail
{



// assumed, that nonlinear operator contains the jacoboan s.t. it can perofrm operator-vector application
template<class VectorOperations, class NonlinearOperator>
class glued_nonlinear_operator_and_jacobian
{
private:
    struct fake_deleter
    {
        template<class PT>
        void operator()(PT* p) const {}
    }; 
public:
    using glued_vector_operations_type = scfd::linspace::glued_vector_space<VectorOperations, 2>;
    using vector_type = typename glued_vector_operations_type::vector_type;
    using scalar_type = typename VectorOperations::scalar_type;
private:
    using T = scalar_type;
    using T_vec = vector_type;

public:

    glued_nonlinear_operator_and_jacobian(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p)
    {

        std::shared_ptr<VectorOperations> aaa(vec_ops_, fake_deleter() );
        glued_vector_operations_ = new glued_vector_operations_type(aaa, aaa);
    }
    ~glued_nonlinear_operator_and_jacobian()
    {
        delete glued_vector_operations_;
    }
    
    glued_vector_operations_type* glued_vector_operations()
    {
        return glued_vector_operations_;
    }

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p)
    {
        nonlin_op_->F(time_p, in_p.comp(0), param_p, out_p.comp(0) );
        nonlin_op_->set_linearization_point(in_p.comp(0), param_p);
        nonlin_op_->jacobian_u( in_p.comp(1), out_p.comp(1) );
    }

    void norm_bifurcation_diagram(const T_vec& v_in, std::vector<T>& bif_norms_at_t_)const 
    {
        nonlin_op_->norm_bifurcation_diagram(v_in.comp(0), bif_norms_at_t_);
    }
    T check_solution_quality(const T_vec& v_out)const
    {
        return nonlin_op_->check_solution_quality(v_out.comp(0));
    }

    glued_vector_operations_type* get_glued_vec_ops()const
    {
        return glued_vector_operations_;
    }

protected:
    glued_vector_operations_type* glued_vector_operations_;
private:
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;



};

}
}
#endif