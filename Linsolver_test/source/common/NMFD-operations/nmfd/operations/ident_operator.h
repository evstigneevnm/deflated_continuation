#ifndef __NMFD_IDENT_OPERATOR_H__
#define __NMFD_IDENT_OPERATOR_H__

namespace nmfd
{
namespace operations
{

template<class VectorSpace> 
class ident_operator
{
public:    
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    //using ordinal_type = typename VectorSpace::ordinal_type;
public:

    ident_operator(std::shared_ptr<VectorSpace> space) : space_(space)
    {
    }

    const std::shared_ptr<vector_space_type> &get_dom_space()const
    {
        return space_;
    }
    const std::shared_ptr<vector_space_type> &get_im_space()const
    {
        return space_;
    }

    /// inplace version
    void apply(vector_type& x)const
    { 
    }
    /// outofplace version
    void apply(const vector_type& x, vector_type& f)const
    { 
        space_->assign(x,f);
    }

private:
    std::shared_ptr<VectorSpace> space_;
};

}
}

#endif