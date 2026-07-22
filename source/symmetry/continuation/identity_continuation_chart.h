#ifndef __SYMMETRY_CONTINUATION_IDENTITY_CONTINUATION_CHART_H__
#define __SYMMETRY_CONTINUATION_IDENTITY_CONTINUATION_CHART_H__

#include <symmetry/continuation/continuation_chart_state.h>

namespace symmetry
{
namespace continuation
{

template<class VectorOperations>
class identity_continuation_chart
{
public:
    using vector_type = typename VectorOperations::vector_type;
    using state_type = identity_continuation_chart_state;

    explicit identity_continuation_chart(VectorOperations* vec_ops_):
        vec_ops(vec_ops_)
    {
    }

    void prepare_seed(const vector_type& source, vector_type& destination)
    {
        if(&source != &destination)
        {
            vec_ops->assign(source, destination);
        }
        ++state_.generation;
    }

    void accept_step(vector_type&, vector_type&)
    {
        ++state_.generation;
    }

    const state_type& state() const
    {
        return state_;
    }

private:
    VectorOperations* vec_ops;
    state_type state_;
};

} // namespace continuation
} // namespace symmetry

#endif
