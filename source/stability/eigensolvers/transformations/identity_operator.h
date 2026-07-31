#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_IDENTITY_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_IDENTITY_OPERATOR_H__

#include <cstddef>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace>
class identity_operator
{
public:
    using vector_space_type = VectorSpace;
    using vector_type = typename vector_space_type::vector_type;

    explicit identity_operator(const vector_space_type& vector_space)
        : vector_space_(vector_space)
    {
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        vector_space_.assign(source, destination);
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

private:
    const vector_space_type& vector_space_;
    mutable std::size_t operator_calls_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
