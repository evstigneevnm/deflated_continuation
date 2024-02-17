#ifndef __SCFD_RESIDUAL_REGULATIZATION_DUMMY_H__
#define __SCFD_RESIDUAL_REGULATIZATION_DUMMY_H__

namespace numerical_algos {
namespace lin_solvers {
namespace detail {

class residual_regularization_dummy
{
public:
    
    residual_regularization_dummy()
    {}
    ~residual_regularization_dummy()
    {}
    
    template<class VecX>
    void apply(VecX &x) const
    {
    }


};

}
}
}

#endif