#ifndef __SCFD_GLUED_VECTOR_SPACE_H__
#define __SCFD_GLUED_VECTOR_SPACE_H__

#include <array>
#include <memory>
#include "glued_vector_operations.h"

namespace scfd
{
namespace linspace
{

template<class VectorSpace, std::size_t n>
class glued_vector_space : 
    public glued_vector_operations<VectorSpace, n>
{
    using parent_t = glued_vector_operations<VectorSpace, n>;
    using internal_space_t = VectorSpace;
    using internal_vector_t = typename VectorSpace::vector_type;

public:
    using typename parent_t::vector_type;
    using typename parent_t::scalar_type;
    using space_comp_type = internal_space_t;

public:
    template
    <
        class... Args,
        class = 
            typename std::enable_if<detail::ctx_all_of< std::is_same<Args,std::shared_ptr<VectorSpace>>::value... >::value>::type,
        class = 
            typename std::enable_if<sizeof...(Args)==n>::type
    >
    glued_vector_space(Args ...ops) : parent_t(ops...)
    {
    }
    glued_vector_space(const std::array<std::shared_ptr<VectorSpace>,n> &ops) : parent_t(ops)
    {
    }
    glued_vector_space(std::shared_ptr<VectorSpace> ops) : parent_t(ops)
    {
    }

    void init_vector(vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            parent_t::internal_ops_[i]->init_vector(x.comp(i));
        }
    }
    void free_vector(vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            parent_t::internal_ops_[i]->free_vector(x.comp(i));
        }
    }
    void start_use_vector(vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            parent_t::internal_ops_[i]->start_use_vector(x.comp(i));
        }        
    }
    void stop_use_vector(vector_type& x) const
    {
        for (std::size_t i = 0;i < n;++i)
        {
            parent_t::internal_ops_[i]->stop_use_vector(x.comp(i));
        }  
    }


    internal_space_t &space_comp(std::size_t i)
    {
        return *parent_t::internal_ops_[i];
    }
    const internal_space_t &space_comp(std::size_t i) const
    {
        return *parent_t::internal_ops_[i];
    }
};

} // namespace linspace
} // namespace scfd

#endif
