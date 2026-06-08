#ifndef __NMFD_DENSE1_EXTENDED_OPERATOR_H__
#define __NMFD_DENSE1_EXTENDED_OPERATOR_H__

#include <cmath>
#include <array>
#include <iterator>
#include <algorithm>
#include <vector>
#include <memory>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>
#include "nmfd/detail/vector_wrap.h"

/// TODO systemize size_t behaviour - mb make some Vector dependent type?
/// TODO multivector - add some generic wrap for multivector (std::vector) and use it here

namespace nmfd
{
namespace operations
{

template<class OrigOperator, class OrigVectorSpace>
class dense1_extended_operator
/*
Class represents operator of form
    (A   u)
    (v^T w)
where A is some origin operator extended with vector u row v^t and number w.
*/
{
    ///TODO add static assert for scalar_type and vector_type coincidence
    static_assert(std::is_same<typename OrigOperator::scalar_type, typename OrigVectorSpace::scalar_type>::value, "scalar_type in OrigOperator and OrigVectorSpace must be the same");
    static_assert(std::is_same<typename OrigOperator::vector_type, typename OrigVectorSpace::vector_type>::value, "vector_type in OrigOperator and OrigVectorSpace must be the same");

    using scalar_type = typename OrigOperator::scalar_type;

    using orig_vector_space_type = OrigVectorSpace; // vector space for u and v^t
    using orig_vector_type = typename OrigOperator::vector_type; // vector type of u and v^t

    using scalar_space_type = operations::static_vector_space<scalar_type,1>; // vector space for w
    using scalar_vector_type = typename scalar_space_type::vector_type; // vector type of w

    using vector_space_type = operations::pair_vector_space<OrigVectorSpace,scalar_space_type>;
    using vector_type = typename vector_space_type::vector_type;

public:
    dense1_extended_operator(std::shared_ptr<OrigVectorSpace> orig_vec_space, std::shared_ptr<const OrigOperator> orig_op = nullptr):
        orig_vec_space_(orig_vec_space),
        scalar_space_(std::make_shared<scalar_space_type>()),
        pair_space_(std::make_shared<vector_space_type>(orig_vec_space, scalar_space_)),
        A_(orig_op),
        u_wrap_(*orig_vec_space),
        v_wrap_(*orig_vec_space),
        w_wrap_(*scalar_space_)
    {
    }

    dense1_extended_operator(std::shared_ptr<OrigVectorSpace> orig_vec_space, std::shared_ptr<const OrigOperator> orig_op, const orig_vector_type &_u, const orig_vector_type &_v, const scalar_vector_type &_w):
        dense1_extended_operator(orig_vec_space, orig_op)
    {
        orig_vec_space_->assign(_u, *u_wrap_);
        orig_vec_space_->assign(_v, *v_wrap_);
        scalar_space_->assign(_w, *w_wrap_);
    }

    /// Use to set or reset origin A operator
    void set_orig_operator(std::shared_ptr<const OrigOperator> orig_op)
    {
        A_ = orig_op;
    }

    std::shared_ptr<const OrigOperator> get_orig_operator()
    {
        return A_;
    }

    const std::shared_ptr<const OrigOperator> get_orig_operator() const
    {
        return A_;
    }

    // TODO: Add const versions of u() v() w()?
    ///These can be used to reset dense extension values
    orig_vector_type &u()
    {
        return *u_wrap_;
    }

    orig_vector_type &v()
    {
        return *v_wrap_;
    }

    scalar_vector_type &w()
    {
        return *w_wrap_;
    }

    const orig_vector_type &u() const
    {
        return *u_wrap_;
    }

    const orig_vector_type &v() const
    {
        return *v_wrap_;
    }

    const scalar_vector_type &w() const
    {
        return *w_wrap_;
    }

    void apply(const vector_type &in, vector_type &out)
    {
        const scalar_type scalar_in_second = scalar_space_->get_value_at_point(0, in.second);
        const scalar_type scalar_w = scalar_space_->get_value_at_point(0, *w_wrap_);

        // out.first() = A.apply(in.first()) + in.second()*u
        A_->apply(in.first, out.first);
        orig_vec_space_->add_lin_comb(scalar_in_second, *u_wrap_, out.first);

        // out.second() = v^t * in.first() + w * in.second()
        scalar_type scalar_v_in_first = orig_vec_space_->scalar_prod(*v_wrap_, in.first);
        scalar_type scalar_w_in_second = scalar_space_->scalar_prod(*w_wrap_, in.second);
        scalar_space_->set_value_at_point(scalar_v_in_first + scalar_w_in_second, 0, out.second);
    }

    // TODO: Is it correct?
    const std::shared_ptr<vector_space_type>& get_im_space() const
    {
        return pair_space_;
    }

    // TODO: Is it correct?
    const std::shared_ptr<vector_space_type>& get_dom_space() const
    {
        return pair_space_;
    }


private:
    using orig_vector_wrap_t = detail::vector_wrap<orig_vector_space_type, true, true>;
    using scalar_vector_wrap_t = detail::vector_wrap<scalar_space_type, true, true>;

    std::shared_ptr<orig_vector_space_type> orig_vec_space_;
    std::shared_ptr<scalar_space_type> scalar_space_;
    std::shared_ptr<vector_space_type> pair_space_;

    std::shared_ptr<const OrigOperator> A_;
    orig_vector_wrap_t u_wrap_;
    orig_vector_wrap_t v_wrap_;
    scalar_vector_wrap_t w_wrap_;

};

} // namespace operations
} // namespace nmfd

#endif
