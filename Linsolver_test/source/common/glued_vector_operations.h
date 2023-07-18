#ifndef __SCFD_GLUED_VECTOR_OPERATIONS_H__
#define __SCFD_GLUED_VECTOR_OPERATIONS_H__

#include <array>
#include <memory>
#include "detail/ctx_all_of.h"
#include "vector_operations_base.h"

namespace scfd
{
namespace linspace
{

template<class Vector, std::size_t n>
class glued_vector
{
    std::array<Vector,n> internal_vectors_;
public:
    Vector &comp(std::size_t comp_i)
    {
        return internal_vectors_[comp_i];
    }
    const Vector &comp(std::size_t comp_i)const
    {
        return internal_vectors_[comp_i];
    }
};


template<class VectorOperations, std::size_t n>
class glued_vector_operations : 
    public vector_operations_base
    <
        typename VectorOperations::scalar_type, 
        glued_vector<typename VectorOperations::vector_type,n>
    >
{
    using parent_t = 
        vector_operations_base
        <
            typename VectorOperations::scalar_type, 
            glued_vector<typename VectorOperations::vector_type,n>
        >;
    using internal_ops_t = VectorOperations;
    using internal_vector_t = typename VectorOperations::vector_type;

public:
    using typename parent_t::vector_type;
    using typename parent_t::scalar_type;

public:
    template
    <
        class... Args,
        class = 
            typename std::enable_if<detail::ctx_all_of< std::is_same<Args,std::shared_ptr<VectorOperations>>::value... >::value>::type,
        class = 
            typename std::enable_if<sizeof...(Args)==n>::type
    >
    glued_vector_operations(Args ...ops) : internal_ops_{ops...}
    { }
    glued_vector_operations(const std::array<std::shared_ptr<VectorOperations>,n> &ops) : internal_ops_(ops)
    {
    }
    glued_vector_operations(std::shared_ptr<VectorOperations> ops)
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i] = ops;
        }
    }


    [[nodiscard]] bool check_is_valid_number(const vector_type &x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            if (!internal_ops_[i]->check_is_valid_number(x.comp(i))) return false;
        }
        return true;
    }

    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type res(0);
        for (std::size_t i = 0;i < n;++i)
        {
            res += internal_ops_[i]->scalar_prod(x.comp(i),y.comp(i));
        }
        return res;
    }

    [[nodiscard]] scalar_type sum(const vector_type &x)const
    {
        scalar_type res(0);
        for (std::size_t i = 0;i < n;++i)
        {
            res += internal_ops_[i]->sum(x.comp(i));
        }
        return res;
    }
    
    [[nodiscard]] scalar_type asum(const vector_type &x)const
    {
        scalar_type res(0);
        for (std::size_t i = 0;i < n;++i)
        {
            res += internal_ops_[i]->asum(x.comp(i));
        }
        return res;
    }

    [[nodiscard]] scalar_type norm(const vector_type &x) const
    {
        return std::sqrt(norm_sq(x));
    }
    [[nodiscard]] scalar_type norm2(const vector_type &x) const
    {
        return std::sqrt(norm2_sq(x));
    }
    [[nodiscard]] scalar_type norm_sq(const vector_type &x) const
    {
        scalar_type res(0);
        for (std::size_t i = 0;i < n;++i)
        {
            res += internal_ops_[i]->norm_sq(x.comp(i));
        }
        return res;
    }
    [[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
    {
        scalar_type res(0);
        for (std::size_t i = 0;i < n;++i)
        {
            res += internal_ops_[i]->norm2_sq(x.comp(i));
        }
        return res;
    }
    void make_abs_copy(const vector_type& x, vector_type& y)const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->make_abs_copy( x.comp(i), y.comp(i) );
        }        
    }
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->max_pointwise( sc, x.comp(i), y.comp(i) );
        }         
    }
    void max_pointwise(const scalar_type sc, vector_type& y)const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->max_pointwise(sc, y.comp(i) );
        } 
    }     
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, 
                        vector_type& z)const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->div_pointwise( mul_x, x.comp(i), mul_y, y.comp(i), z.comp(i) );
        }          
    }
    scalar_type norm_inf(const vector_type& x)const
    {
        scalar_type ret_norm = 0;
        for (std::size_t i = 0;i < n;++i)
        {
            ret_norm = std::max(internal_ops_[i]->norm_inf( x.comp(i) ), ret_norm);
        }  
        return ret_norm;
    }
    void assign_scalar(const scalar_type scalar, vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->assign_scalar(scalar,x.comp(i));
        }
    }
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->add_mul_scalar(scalar,mul_x,x.comp(i));
        }
    }
    void scale(const scalar_type scale, vector_type &x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->scale(scale,x.comp(i));
        }
    }
    void assign(const vector_type& x, vector_type& y) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->assign(x.comp(i),y.comp(i));
        }
    }
    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->assign_mul(mul_x,x.comp(i),y.comp(i));
        }
    }
    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->add_mul(mul_x,x.comp(i),mul_y,y.comp(i));
        }
    }
    void add_mul(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_[i]->add_mul(mul_x,x.comp(i), y.comp(i));
        }
    }

protected:
    std::array<std::shared_ptr<VectorOperations>,n> internal_ops_;
};

} // namespace linspace
} // namespace scfd

#endif
