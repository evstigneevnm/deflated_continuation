#ifndef __NMFD_RECT_VECTOR_SPACE_H__
#define __NMFD_RECT_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <vector>

#include <nmfd/operations/vector_operations_base.h>
#include <nmfd/operations/vector_space_base.h>
#include <nmfd/operations/default_multivector_space_base.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/vec.h>
#include <scfd/arrays/tensorN_array_nd.h>

#include "kernels/rect_vector_space.h"

namespace nmfd
{

template
<
    class Type,
    int   Dim,
    int   TensorDim,
    class Backend,
    class Ordinal=std::ptrdiff_t,
    /***********************************************************/
    class VectorType=scfd::arrays::tensor1_array_nd<Type, Dim, typename Backend::memory_type, TensorDim>,
    class IdxType=scfd::static_vec::vec<Ordinal, Dim>
>
class rect_vector_space :
    public nmfd::operations::default_multivector_space_base
    <
        rect_vector_space<Type, Dim, TensorDim, Backend, Ordinal>,
        Type, VectorType, Ordinal
    >
{
    using parent_t = nmfd::operations::default_multivector_space_base
    <
        rect_vector_space<Type, Dim, TensorDim, Backend, Ordinal>,
        Type, VectorType, Ordinal
    >;

public:
    static const int dim         = Dim;
    static const int tensor_dim  = TensorDim;
    using value_type             = Type;
    using scalar_type            = Type;
    using ordinal_type           = Ordinal;

    using backend_type           = Backend;

    using for_each_nd_type       = typename Backend::template for_each_nd_type<Dim, Ordinal>;
    using reduce_type            = typename Backend::reduce_type;
    using memory_type            = typename Backend::memory_type;

    using multivector_type       = typename parent_t::multivector_type;
    using vector_type            = VectorType;
    using array_nd_type          = scfd::arrays::array_nd<Type, Dim, typename Backend::memory_type>;
    using idx_nd_type            = IdxType;

public: // Especially for SYCL
    using shur_prod_kernel       = kernels::shur_prod<idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
    using sum_kernel             = kernels::sum<idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
    using assign_scalar_kernel   = kernels::assign_scalar<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using add_mul_scalar_kernel  = kernels::add_mul_scalar<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using scale_kernel           = kernels::scale<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_kernel          = kernels::assign<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_lin_comb_1_kernel = kernels::assign_lin_1_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_lin_comb_2_kernel = kernels::assign_lin_2_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using add_lin_comb_kernel    = kernels::add_lin_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;

private:
    idx_nd_type                 range;
    ordinal_type                   sz;
    array_nd_type mutable  helper;

    for_each_nd_type for_each_nd_inst;
    reduce_type           reduce_inst;

public:
    rect_vector_space(idx_nd_type const r, bool use_high_precision = false):
        parent_t(use_high_precision), range(r), sz(r.components_prod() * tensor_dim), helper(range) {};
    //sz is total size meanwhile range is vector space size
    idx_nd_type get_size() const noexcept { return range; }
    idx_nd_type size()     const noexcept { return range; }
public:
    void init_vector(vector_type& x) const override { x.init(range); }
    void free_vector(vector_type& x) const override { x.free();      }

    void start_use_vector(vector_type& x) const override {}
    void  stop_use_vector(vector_type& x) const override {}

public: // Implementing Vector_Operations interface
    [[nodiscard]] bool is_valid_number(const vector_type &x) const override
    {
        return std::isfinite(sum(x));
        //throw std::logic_error("device_vector_space: multivector.is_valid_number not implemented yet");
        //return true; //TODO
    }

    // reduction operations:
    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, y, helper}, range);
        return reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0});
    }

    [[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y) const override
    {
        return scalar_prod(x, y);
    }

    // multivector interface implementations from parent restore because of overshadowing
    using parent_t::assign;
    using parent_t::scalar_prod;
    using parent_t::scalar_prod_l2;
    using parent_t::add_lin_comb;

    [[nodiscard]] scalar_type sum(const vector_type &x) const override
    {
        for_each_nd_inst(sum_kernel{x, helper}, range);
        return reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0});
    }

    [[nodiscard]] scalar_type asum(const vector_type &x) const override
    {
        throw std::logic_error("rect_vector_space: multivector.asum not implemented yet");
        return 0; //TODO
    }

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] scalar_type norm(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0}));
    }

    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] scalar_type norm_l2(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0}) / sz);
    }

    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] scalar_type norm_sq(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0});
    }

    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] scalar_type norm2_sq(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(range.components_prod(), helper.raw_ptr(), scalar_type{0}) / sz;
    }

public:
    //calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar(const scalar_type scalar, vector_type& x) const override
    {
        for_each_nd_inst(assign_scalar_kernel{scalar, x}, range);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const override
    {
        for_each_nd_inst(add_mul_scalar_kernel{scalar, mul_x, x}, range);
    }

    //calc: x := scale*x
    void scale(const scalar_type scale, vector_type &x) const override
    {
        for_each_nd_inst(scale_kernel{scale, x}, range);
    }

    //copy: y := x
    void assign(const vector_type& x, vector_type& y) const override
    {
        for_each_nd_inst(assign_kernel{x, y}, range);
    }

    //calc: y := mul_x*x
    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const override
    {
        for_each_nd_inst(assign_lin_comb_1_kernel{mul_x, x, y}, range);
    }

    //calc: z := mul_x*x + mul_y*y
    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const override
    {
        for_each_nd_inst(assign_lin_comb_2_kernel{mul_x, mul_y, x, y, z}, range);
    }

    //calc: y := mul_x*x + mul_y*y
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const override
    {
        for_each_nd_inst(add_lin_comb_kernel{mul_x, mul_y, x, y}, range);
    }
};

}// namespace nmfd

#endif
