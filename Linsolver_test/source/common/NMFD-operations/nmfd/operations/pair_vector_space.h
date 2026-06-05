#ifndef __NMFD_PAIR_VECTOR_SPACE_H__
#define __NMFD_PAIR_VECTOR_SPACE_H__

#include <cmath>
#include <array>
#include <iterator>
#include <algorithm>
#include <vector>
#include <memory>

/// TODO systemize size_t behaviour - mb make some Vector dependent type?
/// TODO multivector - add some generic wrap for multivector (std::vector) and use it here

namespace nmfd
{
namespace operations
{

template <class VectorSpace1, class VectorSpace2>
struct pair_vector_space
{
    using scalar1_type = typename VectorSpace1::scalar_type;
    using scalar2_type = typename VectorSpace2::scalar_type;
    static_assert(std::is_same<scalar1_type, scalar2_type>::value, "scalar1_type and scalar2_type must be the same");
    using scalar_type = scalar1_type;

    using vector1_type = typename VectorSpace1::vector_type;
    using vector2_type = typename VectorSpace2::vector_type;
    using vector_type = std::pair<vector1_type,vector2_type>;

public:
    pair_vector_space(std::shared_ptr<VectorSpace1> vs1_, std::shared_ptr<VectorSpace2> vs2_)
        : vs1(vs1_), vs2(vs2_)
    {
    }

    std::shared_ptr<VectorSpace1> first()
    {
        return vs1;
    }

    std::shared_ptr<VectorSpace2> second()
    {
        return vs2;
    }

    const std::shared_ptr<VectorSpace1>& first() const
    {
        return vs1;
    }

    const std::shared_ptr<VectorSpace2>& second() const
    {
        return vs2;
    }

    size_t size()const
    {
        return vs1->size() + vs2->size();
    }
    size_t get_size(const vector_type& x)const
    {
        return vs1->get_size(x.first) + vs2->get_size(x.second);
    }


    void init_vector(vector_type& x)const
    {
        vs1->init_vector(x.first);
        vs2->init_vector(x.second);
    }
    template<class ...Args>
    void init_vectors(Args&&...args)const
    {
        // TODO: implement
    }
    void free_vector(vector_type& x)const
    {
        vs1->free_vector(x.first);
        vs2->free_vector(x.second);
    }
    template<class ...Args>
    void free_vectors(Args&&...args) const
    {
        // TODO: implement
    }
    void start_use_vector(vector_type& x)const
    {
        vs1->start_use_vector(x.first);
        vs2->start_use_vector(x.second);
    }
    template<class ...Args>
    void start_use_vectors(Args&&...args)const
    {
        // TODO: implement
    }
    void stop_use_vector(vector_type& x)const
    {
        vs1->stop_use_vector(x.first);
        vs2->stop_use_vector(x.second);
    }
    template<class ...Args>
    void stop_use_vectors(Args&&...args)const
    {
        // TODO: implement
    }
    bool check_is_valid_number(const vector_type &x)const
    {
        return vs1->check_is_valid_number(x.first) && vs2->check_is_valid_number(x.second);
    }


    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type res = vs1->scalar_prod(x.first, y.first);
        res += vs2->scalar_prod(x.second, y.second);

        return res;
    }

    [[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const
    {
        return scalar_prod(x, y);
    }

    [[nodiscard]] scalar_type norm(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    [[nodiscard]] scalar_type norm_sq(const vector_type &x)const
    {
        return scalar_prod(x, x);
    }
    [[nodiscard]] scalar_type norm2_sq(const vector_type& x)const
    {
        return norm_sq(x);
    }
    [[nodiscard]] scalar_type norm2(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    [[nodiscard]] scalar_type norm_l2_sq(const vector_type& x)const
    {
        return norm_sq(x);
    }
    [[nodiscard]] scalar_type norm_l2(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    // asum(x)
    [[nodiscard]] scalar_type norm1(const vector_type &x) const
    {
        return asum(x);
    }
    // returns some weighted L1/l1 norm (problem dependent)
    [[nodiscard]] scalar_type norm_l1(const vector_type &x) const
    {
        return norm1(x);
    }
    scalar_type norm_inf(const vector_type& x)const
    {
        scalar_type max_val1 = vs1->norm_inf(x.first);
        scalar_type max_val2 = vs2->norm_inf(x.second);

        return (max_val1<max_val2)?max_val2:max_val1;
    }
    // returns maximum of all elements absolute weighted values (problem dependent)
    [[nodiscard]] scalar_type norm_l_inf(const vector_type &x) const
    {
        return norm_inf(x);
    }

    [[nodiscard]] scalar_type sum(const vector_type &x)const
    {
        return vs1->sum(x.first) + vs2->sum(x.second);
    }
    [[nodiscard]] scalar_type asum(const vector_type &x)const
    {
        return vs1->asum(x.first) + vs2->asum(x.second);
    }

    scalar_type normalize(vector_type& x)const
    {
        scalar_type norm_x = norm(x);
        if (norm_x > 0.0)
        {
            scale(static_cast<scalar_type>(1.0) / norm_x, x);
        }

        return norm_x;
    }

    // TODO: How to assign scalars?
    void set_value_at_point(scalar_type val_x, size_t at, vector_type& x) const
    {
        if (0 <= at && at < vs1->size())
        {
            vs1->set_value_at_point(val_x, at, x.first);
        }
        else if (at < vs1->size() + vs2->size())
        {
            vs2->set_value_at_point(val_x, at - vs1->size(), x.second);
        }
        else
        {
            throw std::logic_error("pair_vector_space::set_value_at_point: at index is out of range");
        }
    }

    scalar_type get_value_at_point(size_t at, const vector_type& x) const
    {
        if (at < vs1->size())
        {
            return vs1->get_value_at_point(at, x.first);
        }
        else if (at < vs1->size() + vs2->size())
        {
            return vs2->get_value_at_point(at - vs1->size(), x.second);
        }
        throw std::logic_error("pair_vector_space::get_value_at_point: at index is out of range");
    }

    //calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar(const scalar_type scalar, vector_type& x)const
    {
        vs1->assign_scalar(scalar, x.first);
        vs2->assign_scalar(scalar, x.second);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const
    {
        vs1->add_mul_scalar(scalar, mul_x, x.first);
        vs2->add_mul_scalar(scalar, mul_x, x.second);
    }

    //calc: x := scale*x
    void scale(scalar_type scale, vector_type &x)const
    {
        add_mul_scalar(scalar_type(0.0), scale, x);
    }

    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        vs1->assign(x.first, y.first);
        vs2->assign(x.second, y.second);
    }
    //calc: y := mul_x*x
    void assign_lin_comb(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        vs1->assign_lin_comb(mul_x, x.first, y.first);
        vs2->assign_lin_comb(mul_x, x.second, y.second);
    }
    //calc: z := mul_x*x + mul_y*y
    void assign_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, vector_type& z)const
    {
        vs1->assign_lin_comb(mul_x, x.first, mul_y, y.first, z.first);
        vs2->assign_lin_comb(mul_x, x.second, mul_y, y.second, z.second);
    }

    //calc: y := mul_x*x + y
    void add_lin_comb(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        vs1->add_lin_comb(mul_x, x.first, y.first);
        vs2->add_lin_comb(mul_x, x.second, y.second);
    }
    //calc: y := mul_x*x + mul_y*y
    void add_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        vs1->add_lin_comb(mul_x, x.first, mul_y, y.first);
        vs2->add_lin_comb(mul_x, x.second, mul_y, y.second);
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y,
                            scalar_type mul_z, vector_type& z)const
    {
        vs1->add_lin_comb(mul_x, x.first, mul_y, y.first, mul_z, z.first);
        vs2->add_lin_comb(mul_x, x.second, mul_y, y.second, mul_z, z.second);
    }

    void make_abs_copy(const vector_type& x, vector_type& y)const
    {
        vs1->make_abs_copy(x.first, y.first);
        vs2->make_abs_copy(x.second, y.second);
    }
    void make_abs(vector_type& x)const
    {
        vs1->make_abs(x.first);
        vs2->make_abs(x.second);
    }

    // y_j = max(x_j,y_j,sc)
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        vs1->max_pointwise(sc, x.first, y.first);
        vs2->max_pointwise(sc, x.second, y.second);
    }
    void max_pointwise(const scalar_type sc, vector_type& y)const
    {
        vs1->max_pointwise(sc, y.first);
        vs2->max_pointwise(sc, y.second);
    }
    // y_j = min(x_j,y_j,sc)
    void min_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        vs1->min_pointwise(sc, x.first, y.first);
        vs2->min_pointwise(sc, x.second, y.second);
    }
    void min_pointwise(const scalar_type sc, vector_type& y)const
    {
        vs1->min_pointwise(sc, y.first);
        vs2->min_pointwise(sc, y.second);
    }

    // //calc: x := x*mul_y*y
    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        vs1->mul_pointwise(x.first, mul_y, y.first);
        vs2->mul_pointwise(x.second, mul_y, y.second);
    }
    //calc: z := mul_x*x*mul_y*y
    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                        vector_type& z)const
    {
        vs1->mul_pointwise(mul_x, x.first, mul_y, y.first, z.first);
        vs2->mul_pointwise(mul_x, x.second, mul_y, y.second, z.second);
    }
    //calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                        vector_type& z)const
    {
        vs1->div_pointwise(mul_x, x.first, mul_y, y.first, z.first);
        vs2->div_pointwise(mul_x, x.second, mul_y, y.second, z.second);
    }
    //calc: x := x/(mul_y*y)
    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        vs1->div_pointwise(x.first, mul_y, y.first);
        vs2->div_pointwise(x.second, mul_y, y.second);
    }

    // //TODO:!
    // /*std::pair<scalar_type, size_t> max_argmax_element(vector_type& y) const
    // {
    //     auto max_iterator = std::max_element(y.begin(), y.end());
    //     size_t argmax = std::distance(y.begin(), max_iterator);

    //     return {*max_iterator, argmax};
    // }

    // scalar_type max_element(vector_type& x)const
    // {
    //     auto ret = max_argmax_element(x);
    //     return ret.first;
    // }

    // size_t argmax_element(vector_type& x)const
    // {
    //     auto ret = max_argmax_element(x);
    //     return ret.second;
    // }*/


    void assign_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > slices, vector_type& y) const
    {
        size_t index_y = 0;
        for(auto& slice: slices)
        {
            size_t begin = slice.first;
            size_t end = slice.second;
            if(end>this->size())
            {
                throw std::logic_error("pair_vector_space::assign_slice: provided slice size is greater than input vector size.");
            }
            for(size_t j = begin; j<end;j++)
            {
                this->set_value_at_point(this->get_value_at_point(j, x), index_y++, y);
            }
        }
    }

    void assign_skip_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > skip_slices, vector_type& y) const
    {
        size_t index_y = 0;
        for(size_t j = 0; j<this->size();j++)
        {
            for(auto& slice: skip_slices)
            {
                size_t begin = slice.first;
                size_t end = slice.second;
                if((j<=begin)||(j>end))
                {
                    this->set_value_at_point(this->get_value_at_point(j, x), index_y++, y);
                }
            }
        }
    }

private:
    std::shared_ptr<VectorSpace1> vs1;
    std::shared_ptr<VectorSpace2> vs2;

};

} // namespace operations
} // namespace nmfd

#endif
