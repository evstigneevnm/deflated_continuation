#ifndef __SCFD_VECTOR_OPERATIONS_LOCAL_H__
#define __SCFD_VECTOR_OPERATIONS_LOCAL_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <scfd/arrays/array.h>
#include <scfd/utils/device_tag.h>

#include <common/scfd_backend_ext/arg_reduce.h>
#include <common/scfd_backend_ext/host_view.h>
#include <common/scfd_backend_ext/math.h>
#include <common/scfd_backend_ext/random.h>

#include <nmfd/operations/vector_space_base.h>

template<class Backend, class T, class Ordinal = std::ptrdiff_t>
class scfd_vector_operations :
    public nmfd::operations::vector_space_base
    <
        T,
        scfd::arrays::array<T, typename Backend::memory_type>,
        std::vector<scfd::arrays::array<T, typename Backend::memory_type>>,
        Ordinal,
        typename common::scfd_backend_ext::scalar_traits<Backend, T>::real_type
    >
{
public:
    using backend_type = Backend;
    using scalar_type = T;
    using ordinal_type = Ordinal;
    using big_ordinal_type = Ordinal;
    using memory_type = typename backend_type::memory_type;
    using vector_type = scfd::arrays::array<scalar_type, memory_type>;
    using multivector_type = std::vector<vector_type>;
    using scalar_traits = common::scfd_backend_ext::scalar_traits<backend_type, scalar_type>;
    using norm_type = typename scalar_traits::real_type;
    using Tsc = norm_type;
    using vector_type_real = scfd::arrays::array<norm_type, memory_type>;
    using host_view_traits = common::scfd_backend_ext::host_view_traits<vector_type>;
    using vector_host_view_type = typename host_view_traits::view_type;
    using host_vector_type = typename host_view_traits::host_array_type;
    using host_view_traits_real = common::scfd_backend_ext::host_view_traits<vector_type_real>;
    using host_vector_type_real = typename host_view_traits_real::host_array_type;
    using for_each_type = typename backend_type::template for_each_type<ordinal_type>;
    using reduce_type = typename backend_type::reduce_type;
    using copy_type = typename backend_type::copy_type;
    using nmfd_parent_type = nmfd::operations::vector_space_base
    <
        scalar_type,
        vector_type,
        multivector_type,
        ordinal_type,
        norm_type
    >;

    using nmfd_parent_type::add_lin_comb;
    using nmfd_parent_type::assign;
    using nmfd_parent_type::scalar_prod;
    using nmfd_parent_type::scalar_prod_l2;

    explicit scfd_vector_operations(std::size_t sz):
        sz_(sz)
    {
        helper_scalar_.init(static_cast<ordinal_type>(sz_));
        helper_real_.init(static_cast<ordinal_type>(sz_));
        ensure_host_buffer();
    }

    ~scfd_vector_operations()
    {
        release_host_view(false);
        free_owned(helper_scalar_);
        free_owned(helper_real_);
        free_owned(host_buffer_);
        free_owned(real_host_buffer_);
    }

    std::size_t get_default_size() const
    {
        return sz_;
    }

    std::size_t size() const
    {
        return sz_;
    }

    std::size_t get_vector_size() const
    {
        return sz_;
    }

    norm_type get_l2_size() const
    {
        return common::scfd_backend_ext::math<backend_type, norm_type>::sqrt(static_cast<norm_type>(sz_));
    }

    std::size_t get_size(const vector_type& x) const
    {
        return static_cast<std::size_t>(x.size());
    }

    bool device_location() const
    {
        return !memory_type::is_host_visible;
    }

    unsigned int get_fp_prec() const
    {
        return std::numeric_limits<norm_type>::digits10 + 1;
    }

    void use_high_precision()
    {
        high_precision_requested_ = true;
    }

    void set_high_precision()
    {
        high_precision_requested_ = true;
        nmfd_parent_type::set_high_precision();
    }

    void set_regular_precision()
    {
        high_precision_requested_ = false;
        nmfd_parent_type::set_regular_precision();
    }

    bool high_precision_requested() const
    {
        return high_precision_requested_;
    }

    void init_vector(vector_type& x) const override
    {
        init_vector(x, std::size_t(0));
    }

    void init_vector(vector_type& x, const std::size_t sz_p) const
    {
        x = vector_type();
    }

    template<class... Args>
    void init_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)init_vector(std::forward<Args>(args)), 0)...};
    }

    void start_use_vector(vector_type& x) const override
    {
        start_use_vector(x, std::size_t(0));
    }

    void start_use_vector(vector_type& x, const std::size_t sz_p) const
    {
        if(x.is_free())
        {
            const std::size_t sz_l = sz_p > 0 ? sz_p : sz_;
            x.init(static_cast<ordinal_type>(sz_l));
        }
    }

    template<class... Args>
    void start_use_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)start_use_vector(std::forward<Args>(args)), 0)...};
    }

    void stop_use_vector(vector_type&) const override
    {
    }

    template<class... Args>
    void stop_use_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)stop_use_vector(std::forward<Args>(args)), 0)...};
    }

    void free_vector(vector_type& x) const override
    {
        free_owned(x);
    }

    template<class... Args>
    void free_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)free_vector(std::forward<Args>(args)), 0)...};
    }

    void init_multivector(multivector_type& x, ordinal_type m) const override
    {
        x.clear();
        x.reserve(static_cast<std::size_t>(m));
        for(ordinal_type i = 0; i < m; ++i)
        {
            vector_type v;
            init_vector(v);
            x.push_back(std::move(v));
        }
    }

    void start_use_multivector(multivector_type& x, ordinal_type m) const override
    {
        for(ordinal_type i = 0; i < m; ++i)
        {
            start_use_vector(x[static_cast<std::size_t>(i)]);
        }
    }

    void stop_use_multivector(multivector_type&, ordinal_type) const override
    {
    }

    void free_multivector(multivector_type& x, ordinal_type m) const override
    {
        for(ordinal_type i = 0; i < m; ++i)
        {
            free_vector(x[static_cast<std::size_t>(i)]);
        }
    }

    vector_type& at(multivector_type& x, std::size_t m, std::size_t k) const
    {
        if(k >= m)
        {
            throw std::out_of_range("scfd_vector_operations::at");
        }
        return x[k];
    }

    void assign(const multivector_type& mx, ordinal_type m, ordinal_type k, vector_type& x) const override
    {
        assign(mx[checked_multivector_index(m, k)], x);
    }

    void assign(const vector_type& x, multivector_type& mx, ordinal_type m, ordinal_type k) const override
    {
        assign(x, mx[checked_multivector_index(m, k)]);
    }

    scalar_type scalar_prod(const multivector_type& mx, ordinal_type m, ordinal_type k, const vector_type& y) const override
    {
        return scalar_prod(mx[checked_multivector_index(m, k)], y);
    }

    scalar_type scalar_prod_l2(const multivector_type& mx, ordinal_type m, ordinal_type k, const vector_type& y) const override
    {
        return scalar_prod_l2(mx[checked_multivector_index(m, k)], y);
    }

    void add_lin_comb(const scalar_type mul_x, const multivector_type& mx, ordinal_type m, ordinal_type k, const scalar_type mul_y, vector_type& y) const override
    {
        add_lin_comb(mul_x, mx[checked_multivector_index(m, k)], mul_y, y);
    }

    scalar_type* view(vector_type& x) const
    {
        open_host_view(x, true);
        return active_host_view_.raw_ptr();
    }

    const scalar_type* view(const vector_type& x) const
    {
        open_host_view(x, true);
        return active_host_view_.raw_ptr();
    }

    scalar_type* get_buffer() const
    {
        ensure_host_buffer();
        return host_buffer_.raw_ptr();
    }

    void set(vector_type& x) const
    {
        if(host_view_active_ && active_host_view_array_ptr_ == x.raw_ptr())
        {
            active_host_view_.sync_to_array();
            release_host_view(false);
            return;
        }
        ensure_host_buffer();
        set(host_buffer_.raw_ptr(), x);
    }

    void set(const scalar_type* host, vector_type& x) const
    {
        copy_type()(static_cast<ordinal_type>(sz_), host, x.raw_ptr());
    }

    void set(const scalar_type* host, vector_type& x, std::size_t n) const
    {
        copy_type()(static_cast<ordinal_type>(n), host, x.raw_ptr());
    }

    void get(const vector_type& x, scalar_type* host) const
    {
        copy_type()(static_cast<ordinal_type>(sz_), x.raw_ptr(), host);
    }

    void get(const vector_type& x, scalar_type* host, std::size_t n) const
    {
        copy_type()(static_cast<ordinal_type>(n), x.raw_ptr(), host);
    }

    bool check_is_valid_number(const vector_type& x) const
    {
        const auto xp = x.raw_ptr();
        auto hp = helper_real_.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            hp[i] = scalar_traits::is_finite(xp[i]) ? norm_type(0) : norm_type(1);
        }, static_cast<ordinal_type>(sz_));
        for_each_.wait();
        return reduce_type()(static_cast<ordinal_type>(sz_), hp, norm_type(0)) == norm_type(0);
    }

    bool is_valid_number(const vector_type& x) const
    {
        return check_is_valid_number(x);
    }

    scalar_type scalar_prod(const vector_type& x, const vector_type& y) const
    {
        const auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        auto hp = helper_scalar_.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            hp[i] = scalar_traits::conj_mul(xp[i], yp[i]);
        }, static_cast<ordinal_type>(sz_));
        for_each_.wait();
        return reduce_type()(static_cast<ordinal_type>(sz_), hp, scalar_traits::zero());
    }

    scalar_type scalar_prod_l2(const vector_type& x, const vector_type& y) const override
    {
        return scalar_prod(x, y);
    }

    scalar_type sum(const vector_type& x) const
    {
        return reduce_type()(static_cast<ordinal_type>(sz_), x.raw_ptr(), scalar_traits::zero());
    }

    norm_type asum(const vector_type& x) const
    {
        const auto xp = x.raw_ptr();
        auto hp = helper_real_.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            hp[i] = scalar_traits::asum_term(xp[i]);
        }, static_cast<ordinal_type>(sz_));
        for_each_.wait();
        return reduce_type()(static_cast<ordinal_type>(sz_), hp, norm_type(0));
    }

    norm_type norm(const vector_type& x) const
    {
        return common::scfd_backend_ext::math<backend_type, norm_type>::sqrt(norm_sq(x));
    }

    norm_type norm_sq(const vector_type& x) const
    {
        const auto xp = x.raw_ptr();
        auto hp = helper_real_.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            hp[i] = scalar_traits::norm_sq_term(xp[i]);
        }, static_cast<ordinal_type>(sz_));
        for_each_.wait();
        return reduce_type()(static_cast<ordinal_type>(sz_), hp, norm_type(0));
    }

    norm_type norm_l2(const vector_type& x) const
    {
        return norm(x)/get_l2_size();
    }

    norm_type norm2(const vector_type& x) const
    {
        return norm_l2(x);
    }

    norm_type norm2_sq(const vector_type& x) const
    {
        return norm_sq(x)/static_cast<norm_type>(sz_);
    }

    norm_type norm_l2_sq(const vector_type& x) const override
    {
        return norm2_sq(x);
    }

    norm_type norm1(const vector_type& x) const override
    {
        return asum(x);
    }

    norm_type norm_l1(const vector_type& x) const override
    {
        return norm1(x);
    }

    norm_type norm_rank1(const vector_type& x, const scalar_type val_x) const
    {
        return common::scfd_backend_ext::math<backend_type, norm_type>::sqrt(norm_sq(x) + scalar_traits::norm_sq_term(val_x));
    }

    norm_type norm_rank1_l2(const vector_type& x, const scalar_type val_x) const
    {
        return norm_rank1(x, val_x)/get_l2_size();
    }

    norm_type norm_inf(const vector_type& x) const
    {
        const auto xp = x.raw_ptr();
        auto hp = helper_real_.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            hp[i] = scalar_traits::asum_term(xp[i]);
        }, static_cast<ordinal_type>(sz_));
        for_each_.wait();
        return reduce_type()(static_cast<ordinal_type>(sz_), hp, norm_type(0), common::scfd_backend_ext::max_op<norm_type>());
    }

    norm_type norm_l_inf(const vector_type& x) const override
    {
        return norm_inf(x);
    }

    norm_type normalize(vector_type& x) const
    {
        const norm_type norm_x = norm(x);
        if(norm_x > norm_type(0))
        {
            scale(scalar_type(norm_type(1)/norm_x), x);
        }
        return norm_x;
    }

    void assign_scalar(const scalar_type scalar, vector_type& x) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] = scalar;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] = mul_x*xp[i] + scalar;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void scale(const scalar_type alpha, vector_type& x) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] *= alpha;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void assign(const vector_type& x, vector_type& y) const
    {
        copy_type()(static_cast<ordinal_type>(x.size()), x.raw_ptr(), y.raw_ptr());
    }

    void swap(vector_type& x, vector_type& y) const
    {
        auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const scalar_type tmp = xp[i];
            xp[i] = yp[i];
            yp[i] = tmp;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] = mul_x*xp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const override
    {
        assign_mul(mul_x, x, y);
    }

    void assign_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
    {
        const auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        auto zp = z.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            zp[i] = mul_x*xp[i] + mul_y*yp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const override
    {
        assign_mul(mul_x, x, mul_y, y, z);
    }

    void add_mul(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] += mul_x*xp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void add_mul(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] = mul_x*xp[i] + mul_y*yp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        add_mul(mul_x, x, mul_y, y);
    }

    void add_mul(
        const scalar_type mul_x,
        const vector_type& x,
        const scalar_type mul_y,
        const vector_type& y,
        const scalar_type mul_z,
        vector_type& z) const
    {
        const auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        auto zp = z.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            zp[i] = mul_x*xp[i] + mul_y*yp[i] + mul_z*zp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void add_lin_comb(
        const scalar_type mul_x,
        const vector_type& x,
        const scalar_type mul_y,
        const vector_type& y,
        const scalar_type mul_z,
        vector_type& z) const
    {
        add_mul(mul_x, x, mul_y, y, mul_z, z);
    }

    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
    {
        const auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        auto zp = z.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            zp[i] = (mul_x*xp[i])*(mul_y*yp[i]);
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y) const
    {
        auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] *= mul_y*yp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
    {
        const auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        auto zp = z.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            zp[i] = (mul_x*xp[i])/(mul_y*yp[i]);
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y) const
    {
        auto xp = x.raw_ptr();
        const auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] /= (mul_y*yp[i]);
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void make_abs_copy(const vector_type& x, vector_type_real& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] = scalar_traits::asum_term(xp[i]);
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void make_abs(vector_type_real& x) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            xp[i] = xp[i] < norm_type(0) ? -xp[i] : xp[i];
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void max_pointwise(const norm_type sc, const vector_type_real& x, vector_type_real& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const norm_type xy = xp[i] > yp[i] ? xp[i] : yp[i];
            yp[i] = xy > sc ? xy : sc;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void max_pointwise(const norm_type sc, vector_type_real& y) const
    {
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] = yp[i] > sc ? yp[i] : sc;
        }, static_cast<ordinal_type>(y.size()));
        for_each_.wait();
    }

    void min_pointwise(const norm_type sc, const vector_type_real& x, vector_type_real& y) const
    {
        const auto xp = x.raw_ptr();
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const norm_type xy = xp[i] < yp[i] ? xp[i] : yp[i];
            yp[i] = xy < sc ? xy : sc;
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void min_pointwise(const norm_type sc, vector_type_real& y) const
    {
        auto yp = y.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            yp[i] = yp[i] < sc ? yp[i] : sc;
        }, static_cast<ordinal_type>(y.size()));
        for_each_.wait();
    }

    void set_value_at_point(scalar_type val_x, std::size_t at, vector_type& x) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            if(static_cast<std::size_t>(i) == at)
            {
                xp[i] = val_x;
            }
        }, static_cast<ordinal_type>(x.size()));
        for_each_.wait();
    }

    void set_value_at_point(scalar_type val_x, std::size_t at, vector_type& x, std::size_t sz_l) const
    {
        auto xp = x.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            if(static_cast<std::size_t>(i) == at)
            {
                xp[i] = val_x;
            }
        }, static_cast<ordinal_type>(sz_l));
        for_each_.wait();
    }

    scalar_type get_value_at_point(std::size_t at, const vector_type& x) const
    {
        ensure_host_buffer();
        get(x, host_buffer_.raw_ptr());
        return host_buffer_.raw_ptr()[at];
    }

    norm_type max_element(vector_type_real& x) const
    {
        return reduce_type()(static_cast<ordinal_type>(x.size()), x.raw_ptr(), -std::numeric_limits<norm_type>::infinity(), common::scfd_backend_ext::max_op<norm_type>());
    }

    std::size_t argmax_element(vector_type_real& x) const
    {
        return max_argmax_element(x).second;
    }

    std::pair<norm_type, std::size_t> max_argmax_element(vector_type_real& x) const
    {
        return common::scfd_backend_ext::arg_reduce<backend_type, norm_type, ordinal_type>::max_argmax(
            static_cast<ordinal_type>(x.size()),
            x.raw_ptr());
    }

    void assign_slices(
        const vector_type& x,
        const std::vector<std::pair<std::size_t, std::size_t>>& slices,
        vector_type& y) const
    {
        const auto x_size = static_cast<std::size_t>(x.size());
        const auto y_size = static_cast<std::size_t>(y.size());
        std::size_t out_size = 0;
        for(const auto& slice : slices)
        {
            if(slice.first > slice.second || slice.second > x_size)
            {
                throw std::logic_error("scfd_vector_operations::assign_slices: invalid slice range");
            }
            out_size += slice.second - slice.first;
        }
        if(out_size != y_size)
        {
            throw std::logic_error("scfd_vector_operations::assign_slices: slice output size does not match destination vector size");
        }

        std::vector<scalar_type> host_x(x_size);
        std::vector<scalar_type> host_y(y_size);
        get(x, host_x.data(), x_size);
        std::size_t out = 0;
        for(const auto& slice : slices)
        {
            for(std::size_t j = slice.first; j < slice.second; ++j)
            {
                host_y[out++] = host_x[j];
            }
        }
        set(host_y.data(), y, y_size);
    }

    void assign_skip_slices(
        const vector_type& x,
        const std::vector<std::pair<std::size_t, std::size_t>>& skip_slices,
        vector_type& y) const
    {
        const auto x_size = static_cast<std::size_t>(x.size());
        const auto y_size = static_cast<std::size_t>(y.size());
        for(const auto& slice : skip_slices)
        {
            if(slice.first > slice.second || slice.second > x_size)
            {
                throw std::logic_error("scfd_vector_operations::assign_skip_slices: invalid skip range");
            }
        }

        std::vector<scalar_type> host_x(x_size);
        std::vector<scalar_type> host_y(y_size);
        get(x, host_x.data(), x_size);
        std::size_t out = 0;
        for(std::size_t j = 0; j < x_size; ++j)
        {
            bool skip = false;
            for(const auto& slice : skip_slices)
            {
                if(j >= slice.first && j < slice.second)
                {
                    skip = true;
                    break;
                }
            }
            if(!skip)
            {
                if(out >= y_size)
                {
                    throw std::logic_error("scfd_vector_operations::assign_skip_slices: destination vector is too small");
                }
                host_y[out++] = host_x[j];
            }
        }
        if(out != y_size)
        {
            throw std::logic_error("scfd_vector_operations::assign_skip_slices: skip output size does not match destination vector size");
        }
        set(host_y.data(), y, y_size);
    }

    void assign_random(vector_type& vec) const
    {
        const std::size_t seed = random_seed_++;
        common::scfd_backend_ext::random_fill<backend_type, scalar_type, ordinal_type>::uniform01(
            for_each_,
            static_cast<ordinal_type>(vec.size()),
            vec.raw_ptr(),
            seed);
    }

    void assign_random(vector_type& vec, scalar_type a, scalar_type b) const
    {
        assign_random(vec);
        add_mul_scalar(a, b - a, vec);
    }

private:
    static std::size_t checked_multivector_index(ordinal_type m, ordinal_type k)
    {
        if(k < ordinal_type(0) || k >= m)
        {
            throw std::out_of_range("scfd_vector_operations::multivector index");
        }
        return static_cast<std::size_t>(k);
    }

    template<class Array>
    static void free_owned(Array& x)
    {
        if(!x.is_free())
        {
            if(!x.is_own())
            {
                x = Array();
                return;
            }
            x.free();
        }
    }

    void ensure_host_buffer() const
    {
        common::scfd_backend_ext::ensure_host_array(host_buffer_, sz_);
    }

    void open_host_view(const vector_type& x, bool sync_from_array) const
    {
        release_host_view(false);
        active_host_view_.init(x, sync_from_array);
        host_view_active_ = true;
        active_host_view_array_ptr_ = x.raw_ptr();
    }

    void release_host_view(bool sync_to_array) const
    {
        if(!host_view_active_)
        {
            return;
        }
        active_host_view_.release(sync_to_array);
        host_view_active_ = false;
        active_host_view_array_ptr_ = nullptr;
    }

    std::size_t sz_;
    mutable for_each_type for_each_;
    mutable vector_type helper_scalar_;
    mutable vector_type_real helper_real_;
    mutable host_vector_type host_buffer_;
    mutable host_vector_type_real real_host_buffer_;
    mutable vector_host_view_type active_host_view_;
    mutable bool host_view_active_ = false;
    mutable const scalar_type* active_host_view_array_ptr_ = nullptr;
    mutable std::size_t random_seed_ = 1;
    bool high_precision_requested_ = false;
};

#endif
