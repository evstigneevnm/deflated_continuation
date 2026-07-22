#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_CODEC_1D_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_CODEC_1D_H__

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace symmetry
{
namespace fourier
{

enum class real_packed_fourier_1d_layout
{
    interleaved_real_imag
};

template<class VectorOperations, class Complex>
class real_packed_fourier_codec_1d
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using complex_type = Complex;

    real_packed_fourier_codec_1d(
        VectorOperations* vec_ops,
        const std::size_t positive_modes,
        const real_packed_fourier_1d_layout layout =
            real_packed_fourier_1d_layout::interleaved_real_imag):
        vec_ops_(vec_ops),
        positive_modes_(positive_modes),
        layout_(layout),
        host_source_(expected_vector_size(), scalar_type(0)),
        host_destination_(expected_vector_size(), scalar_type(0))
    {
        if(vec_ops_ == nullptr)
        {
            throw std::invalid_argument("real_packed_fourier_codec_1d got null vector operations");
        }
        if(positive_modes_ == 0)
        {
            throw std::invalid_argument("real_packed_fourier_codec_1d needs at least one positive mode");
        }
    }

    std::size_t expected_vector_size() const
    {
        switch(layout_)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                return 2*positive_modes_;
        }
        throw std::runtime_error("real_packed_fourier_codec_1d has an unknown layout");
    }

    void check_vector_size(const vector_type& vector) const
    {
        if(vec_ops_->get_size(vector) != expected_vector_size())
        {
            throw std::runtime_error("real-packed Fourier vector size does not match its layout");
        }
    }

    void unpack(const vector_type& source, std::vector<complex_type>& destination)
    {
        check_vector_size(source);
        if(destination.size() != positive_modes_ + 1)
        {
            throw std::runtime_error("real-packed Fourier spectrum has an unexpected size");
        }

        vec_ops_->get(source, host_source_.data(), host_source_.size());
        std::fill(destination.begin(), destination.end(), complex_type(0));
        switch(layout_)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                for(std::size_t mode = 1; mode <= positive_modes_; ++mode)
                {
                    const std::size_t offset = 2*(mode - 1);
                    destination[mode] =
                        complex_type(host_source_[offset], host_source_[offset + 1]);
                }
                return;
        }
        throw std::runtime_error("real_packed_fourier_codec_1d has an unknown layout");
    }

    void pack(const std::vector<complex_type>& source, vector_type& destination)
    {
        check_vector_size(destination);
        if(source.size() != positive_modes_ + 1)
        {
            throw std::runtime_error("real-packed Fourier spectrum has an unexpected size");
        }

        switch(layout_)
        {
            case real_packed_fourier_1d_layout::interleaved_real_imag:
                for(std::size_t mode = 1; mode <= positive_modes_; ++mode)
                {
                    const std::size_t offset = 2*(mode - 1);
                    host_destination_[offset] = source[mode].real();
                    host_destination_[offset + 1] = source[mode].imag();
                }
                break;
        }
        vec_ops_->set(host_destination_.data(), destination, host_destination_.size());
    }

private:
    VectorOperations* vec_ops_;
    std::size_t positive_modes_;
    real_packed_fourier_1d_layout layout_;
    std::vector<scalar_type> host_source_;
    std::vector<scalar_type> host_destination_;
};

} // namespace fourier
} // namespace symmetry

#endif
