#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <visualization/io/numpy_array_writer.h>

namespace
{

std::uint16_t read_little_endian_u16(std::ifstream& file)
{
    unsigned char bytes[2] = {};
    file.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<std::uint16_t>(bytes[0]) |
        (static_cast<std::uint16_t>(bytes[1]) << 8u);
}

bool validate_double_array(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        return false;
    }
    unsigned char prefix[8] = {};
    file.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    const unsigned char expected[8] = {0x93u, 'N', 'U', 'M', 'P', 'Y', 1u, 0u};
    if(!file || !std::equal(prefix, prefix + 8, expected))
    {
        return false;
    }
    const std::uint16_t header_size = read_little_endian_u16(file);
    std::string header(header_size, '\0');
    file.read(&header[0], static_cast<std::streamsize>(header.size()));
    if(header.find("'descr': '<f8'") == std::string::npos ||
       header.find("'shape': (2, 3)") == std::string::npos)
    {
        return false;
    }
    std::array<double, 6> values{};
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size()*sizeof(double)));
    const std::array<double, 6> expected_values = {
        1.25, -2.5, 3.75, 4.5, -5.25, 6.0};
    return file && values == expected_values;
}

bool validate_float_vector(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        return false;
    }
    file.seekg(8);
    const std::uint16_t header_size = read_little_endian_u16(file);
    std::string header(header_size, '\0');
    file.read(&header[0], static_cast<std::streamsize>(header.size()));
    return header.find("'descr': '<f4'") != std::string::npos &&
        header.find("'shape': (4,)") != std::string::npos;
}

} // namespace

int main()
{
    const auto directory = std::filesystem::temp_directory_path();
    const auto double_path = directory / "deflated_continuation_numpy_writer_double.npy";
    const auto float_path = directory / "deflated_continuation_numpy_writer_float.npy";
    const std::array<double, 6> double_values = {
        1.25, -2.5, 3.75, 4.5, -5.25, 6.0};
    const std::array<float, 4> float_values = {1.0f, 2.0f, 3.0f, 4.0f};

    visualization::io::write_numpy_array(
        double_path,
        double_values.data(),
        std::array<std::size_t, 2>{2, 3});
    visualization::io::write_numpy_array(
        float_path,
        float_values.data(),
        std::array<std::size_t, 1>{4});

    const bool passed = validate_double_array(double_path) &&
        validate_float_vector(float_path) &&
        std::string(visualization::io::numpy_scalar_name<double>()) == "float64" &&
        std::string(visualization::io::numpy_scalar_name<float>()) == "float32";
    std::error_code ignored;
    std::filesystem::remove(double_path, ignored);
    std::filesystem::remove(float_path, ignored);

    if(!passed)
    {
        std::cerr << "NumPy array writer test failed\n";
        return 1;
    }
    std::cout << "NumPy array writer test passed\n";
    return 0;
}
