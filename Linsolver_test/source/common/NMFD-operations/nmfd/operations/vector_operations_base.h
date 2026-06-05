#ifndef __NMFD_VECTOR_OPERATIONS_BASE_H__
#define __NMFD_VECTOR_OPERATIONS_BASE_H__

#include <cstddef>
//#include <map>
#include <utility>
//#include <scfd/utils/logged_obj_base.h>


namespace nmfd
{
namespace operations
{

template<class Type, class VectorType, class MultiVectorType, class Ordinal = std::ptrdiff_t, class NormType = Type>
class vector_operations_base
{

public:
    using vector_type = VectorType;
    using multivector_type = MultiVectorType;
    using scalar_type = Type;
    using norm_type = NormType;
    using Ord = Ordinal;
    using ordinal_type = Ord;
    //using big_ordinal_type = Ord;

private:
    using T = scalar_type;

protected:
    bool use_high_precision_;

public:
    vector_operations_base(bool use_high_precision = false):
      use_high_precision_(use_high_precision)
    {}

    virtual ~vector_operations_base() = default;

    void set_high_precision()
    {
        use_high_precision_ = true;
    }
    void set_regular_precision()
    {
        use_high_precision_ = false;
    }

    // multivector interface
    virtual void assign(const multivector_type& mx, Ord m, Ord k_, vector_type& x) const = 0;
    virtual void assign(const vector_type& x, multivector_type& mx, Ord m, Ord k_) const = 0;
    [[nodiscard]] virtual scalar_type scalar_prod(const multivector_type& mx, Ord m, Ord k_, const vector_type &y)const = 0;
    [[nodiscard]] virtual scalar_type scalar_prod_l2(const multivector_type& mx, Ord m, Ord k_, const vector_type &y)const = 0;
    virtual void add_lin_comb(const scalar_type mul_x, const multivector_type& mx, Ord m, Ord k_, const scalar_type mul_y, vector_type& y) const = 0;

    [[nodiscard]] virtual bool is_valid_number(const vector_type &x) const = 0;
    //reduction operations:
    [[nodiscard]] virtual scalar_type scalar_prod(const vector_type &x, const vector_type &y)const = 0;
    [[nodiscard]] virtual scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const = 0;
    [[nodiscard]] virtual scalar_type sum(const vector_type &x)const = 0;

    [[nodiscard]] virtual norm_type asum(const vector_type &x)const = 0;

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] virtual norm_type norm(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] virtual norm_type norm_l2(const vector_type &x) const = 0;
    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] virtual norm_type norm_sq(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] virtual norm_type norm2_sq(const vector_type &x) const = 0;

    [[nodiscard]] virtual norm_type norm2(const vector_type &x) const
    {
        return norm_l2(x);
    }
    [[nodiscard]] virtual norm_type norm_l2_sq(const vector_type &x) const
    {
        return norm2_sq(x);
    }
    [[nodiscard]] virtual norm_type norm1(const vector_type &x) const
    {
        return asum(x);
    }
    [[nodiscard]] virtual norm_type norm_l1(const vector_type &x) const
    {
        return norm1(x);
    }
    [[nodiscard]] virtual norm_type norm_inf(const vector_type &x) const
    {
        return asum(x);
    }
    [[nodiscard]] virtual norm_type norm_l_inf(const vector_type &x) const
    {
        return norm_inf(x);
    }

    //calc: x := <vector_type with all elements equal to given scalar value>
    virtual void assign_scalar(const scalar_type scalar, vector_type& x) const = 0;
    //calc: x := mul_x*x + <vector_type of all scalar value>
    //ISSUE rename into add_scalar?
    virtual void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const = 0;
    virtual void scale(const scalar_type scale, vector_type &x) const = 0;
    //copy: y := x
    virtual void assign(const vector_type& x, vector_type& y) const = 0;
    //calc: y := mul_x*x
    //ISSUE rename into assign_lin_comb?
    virtual void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const = 0;
    //calc: z := mul_x*x + mul_y*y
    virtual void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const = 0;
    //calc: y := mul_x*x + mul_y*y
    //ISSUE rename into add_lin_comb or add_mul_lin_comb?
    virtual void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const = 0;


};

}
}

#endif
