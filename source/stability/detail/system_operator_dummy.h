#ifndef __STABILITY_SYSTEM_OPERATOR_DUMMY_H__
#define __STABILITY_SYSTEM_OPERATOR_DUMMY_H__

//TODO: put arnoldi to iram_process as a default as well, where this operator is passed there.

namespace stability
{
namespace detail
{
template<class VectorOperations, class LinearOperator>
class system_operator_dummy
{
    using T_vec = typename VectorOperations::vector_type;

public:
    system_operator_dummy(VectorOperations* vec_ops_p, LinearOperator* lin_op_p):
    vec_ops_(vec_ops_p), lin_op_(lin_op_p), target_("LM")
    {}
    ~system_operator_dummy()
    {}
    
    bool solve(const T_vec& v_in, T_vec& v_out)const
    {
        return lin_op_->apply(v_in, v_out);
        
    }
    std::string target_eigs()const
    {
        return target_;
    }
    void set_target_eigs(const std::string& target_p)
    {
        target_ = target_p;
    }

private:
    VectorOperations* vec_ops_;
    LinearOperator* lin_op_;
    std::string target_;
};

}
}


#endif