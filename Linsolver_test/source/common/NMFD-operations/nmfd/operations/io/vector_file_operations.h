#ifndef __NMFD_OPERATIONS_IO_VECTOR_FILE_OPERATIONS_H__
#define __NMFD_OPERATIONS_IO_VECTOR_FILE_OPERATIONS_H__

#include <cstddef>
#include <string>
#include <vector>

#include <nmfd/operations/io/file_operations.h>

namespace nmfd
{
namespace operations
{
namespace io
{

namespace detail
{

template<class View>
auto view_data(View& view) -> decltype(view.raw_ptr())
{
    return view.raw_ptr();
}

template<class View>
auto view_data(const View& view) -> decltype(view.raw_ptr())
{
    return view.raw_ptr();
}

template<class View>
auto view_data(View& view) -> decltype(view.data())
{
    return view.data();
}

template<class View>
auto view_data(const View& view) -> decltype(view.data())
{
    return view.data();
}

template<class T>
T* view_data(T* view)
{
    return view;
}

template<class T>
const T* view_data(const T* view)
{
    return view;
}

template<class VectorOperations, class Vector>
auto mutable_vector_view(int, VectorOperations* vec_ops, Vector& vec, bool sync_from_array, bool sync_to_array_on_destroy)
    -> decltype(vec_ops->view(vec, sync_from_array, sync_to_array_on_destroy))
{
    return vec_ops->view(vec, sync_from_array, sync_to_array_on_destroy);
}

template<class VectorOperations, class Vector>
auto mutable_vector_view(long, VectorOperations* vec_ops, Vector& vec, bool, bool)
    -> decltype(vec_ops->view(vec))
{
    return vec_ops->view(vec);
}

template<class View>
auto release_view(int, View& view, bool sync_to_array) -> decltype(view.release(sync_to_array), void())
{
    view.release(sync_to_array);
}

template<class View>
void release_view(long, View&, bool)
{
}

template<class VectorOperations, class Vector>
auto sync_vector(int, VectorOperations* vec_ops, Vector& vec) -> decltype(vec_ops->set(vec), void())
{
    vec_ops->set(vec);
}

template<class VectorOperations, class Vector>
void sync_vector(long, VectorOperations*, Vector&)
{
}

} // namespace detail

template<class VectorOperations>
class vector_file_operations
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit vector_file_operations(VectorOperations* vec_ops_):
        vec_ops(vec_ops_),
        sz(vec_ops_->get_vector_size()),
        prec(vec_ops_->get_fp_prec())
    {
    }

    void write_vector(const std::string& f_name, const vector_type& vec, unsigned int prec_p = 0) const
    {
        auto host_view = vec_ops->view(vec);
        file_operations::write_vector<scalar_type>(
            f_name,
            sz,
            detail::view_data(host_view),
            prec_p == 0 ? prec : prec_p);
    }

    void read_vector(const std::string& f_name, vector_type& vec) const
    {
        auto host_view = detail::mutable_vector_view(0, vec_ops, vec, false, true);
        file_operations::read_vector<scalar_type>(f_name, sz, detail::view_data(host_view));
        detail::release_view(0, host_view, true);
        detail::sync_vector(0, vec_ops, vec);
    }

    void write_2_vectors_by_side(
        const std::string& f_name,
        const vector_type& vec1,
        const vector_type& vec2,
        unsigned int prec_p = 16,
        char sep = ' ') const
    {
        auto host_view1 = vec_ops->view(vec1);
        auto host_view2 = vec_ops->view(vec2);
        file_operations::write_2_vectors_by_side<scalar_type, const scalar_type*>(
            f_name,
            sz,
            detail::view_data(host_view1),
            detail::view_data(host_view2),
            prec_p,
            sep);
    }

private:
    VectorOperations* vec_ops;
    std::size_t sz;
    unsigned int prec;
};

} // namespace io
} // namespace operations
} // namespace nmfd

#endif
