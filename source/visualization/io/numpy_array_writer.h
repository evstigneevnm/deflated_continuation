#ifndef __VISUALIZATION_IO_NUMPY_ARRAY_WRITER_H__
#define __VISUALIZATION_IO_NUMPY_ARRAY_WRITER_H__

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace visualization
{
namespace io
{

namespace detail
{

template<class T>
struct numpy_scalar_descriptor;

template<>
struct numpy_scalar_descriptor<float>
{
    static const char* dtype() { return "<f4"; }
    static const char* name() { return "float32"; }
};

template<>
struct numpy_scalar_descriptor<double>
{
    static const char* dtype() { return "<f8"; }
    static const char* name() { return "float64"; }
};

inline bool host_is_little_endian()
{
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

template<std::size_t Dimension>
std::size_t checked_element_count(const std::array<std::size_t, Dimension>& shape)
{
    std::size_t result = 1;
    for(const std::size_t extent: shape)
    {
        if(extent == 0)
        {
            throw std::invalid_argument("NumPy array dimensions must be positive");
        }
        if(extent > std::numeric_limits<std::size_t>::max()/result)
        {
            throw std::overflow_error("NumPy array element count overflows size_t");
        }
        result *= extent;
    }
    return result;
}

template<std::size_t Dimension>
std::string shape_tuple(const std::array<std::size_t, Dimension>& shape)
{
    std::ostringstream stream;
    stream << "(";
    for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
    {
        if(dimension != 0)
        {
            stream << ", ";
        }
        stream << shape[dimension];
    }
    if(Dimension == 1)
    {
        stream << ",";
    }
    stream << ")";
    return stream.str();
}

inline void write_little_endian_u16(std::ofstream& file, const std::uint16_t value)
{
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8u) & 0xffu)
    };
    file.write(reinterpret_cast<const char*>(bytes), 2);
}

template<class T>
void write_little_endian_values(std::ofstream& file, const T* values, const std::size_t count)
{
    if(host_is_little_endian())
    {
        file.write(
            reinterpret_cast<const char*>(values),
            static_cast<std::streamsize>(count*sizeof(T)));
        return;
    }

    std::array<unsigned char, sizeof(T)> bytes{};
    for(std::size_t index = 0; index < count; ++index)
    {
        const auto* source = reinterpret_cast<const unsigned char*>(&values[index]);
        std::reverse_copy(source, source + sizeof(T), bytes.begin());
        file.write(reinterpret_cast<const char*>(bytes.data()), sizeof(T));
    }
}

} // namespace detail

template<class T>
const char* numpy_scalar_name()
{
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "NumPy visualization output supports float and double");
    return detail::numpy_scalar_descriptor<T>::name();
}

template<class T, std::size_t Dimension>
void write_numpy_array(
    const std::filesystem::path& path,
    const T* values,
    const std::array<std::size_t, Dimension>& shape)
{
    static_assert(Dimension > 0, "NumPy arrays require at least one dimension");
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "NumPy visualization output supports float and double");

    const std::size_t count = detail::checked_element_count(shape);
    if(values == nullptr && count != 0)
    {
        throw std::invalid_argument("NumPy array data pointer is null");
    }

    std::ostringstream dictionary;
    dictionary << "{'descr': '" << detail::numpy_scalar_descriptor<T>::dtype()
               << "', 'fortran_order': False, 'shape': "
               << detail::shape_tuple(shape) << ", }";

    constexpr std::size_t preamble_size = 10;
    constexpr std::size_t alignment = 16;
    const std::string dictionary_text = dictionary.str();
    const std::size_t unpadded_size = preamble_size + dictionary_text.size() + 1;
    const std::size_t padding = (alignment - unpadded_size%alignment)%alignment;
    const std::string header = dictionary_text + std::string(padding, ' ') + "\n";
    if(header.size() > std::numeric_limits<std::uint16_t>::max())
    {
        throw std::length_error("NumPy v1 header exceeds 65535 bytes");
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if(!file)
    {
        throw std::runtime_error("visualization: failed to open NumPy output file " + path.string());
    }

    const unsigned char magic[] = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const unsigned char version[] = {1u, 0u};
    file.write(reinterpret_cast<const char*>(version), sizeof(version));
    detail::write_little_endian_u16(file, static_cast<std::uint16_t>(header.size()));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    detail::write_little_endian_values(file, values, count);
    file.close();
    if(!file)
    {
        throw std::runtime_error("visualization: failed to write NumPy output file " + path.string());
    }
}

} // namespace io
} // namespace visualization

#endif
