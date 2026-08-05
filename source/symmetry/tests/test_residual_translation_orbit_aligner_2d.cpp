#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

namespace
{

struct vector_operations
{
    using scalar_type = double;
    using norm_type = double;
    using vector_type = std::vector<double>;

    explicit vector_operations(const std::size_t size_): size(size_) {}

    std::size_t get_default_size() const { return size; }
    std::size_t get_size(const vector_type& value) const { return value.size(); }
    void assign(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }
    void get(
        const vector_type& source,
        double* destination,
        const std::size_t count) const
    {
        for(std::size_t index = 0; index < count; ++index)
        {
            destination[index] = source[index];
        }
    }
    void set(
        const double* source,
        vector_type& destination,
        const std::size_t count) const
    {
        destination.assign(source, source + count);
    }

    std::size_t size;
};

void require(const bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

double distance(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    double result = 0.0;
    for(std::size_t index = 0; index < left.size(); ++index)
    {
        const double difference = left[index] - right[index];
        result += difference*difference;
    }
    return std::sqrt(result);
}

} // namespace

int main()
{
    try
    {
        vector_operations operations(3);
        using aligner_type =
            symmetry::fourier::residual_translation_orbit_aligner_2d<
                vector_operations>;
        const std::vector<symmetry::fourier::mode_index<2>> modes{
            {2, 0},
            {4, 0},
            {6, 0}};
        aligner_type aligner(&operations, modes);

        const std::vector<double> reference{2.0, 1.0, 0.5};
        const std::vector<double> translated{-2.0, 1.0, -0.5};
        const std::vector<double> translated_tangent{-0.3, -0.2, -0.1};
        const std::vector<double> expected_tangent{0.3, -0.2, 0.1};
        std::vector<double> aligned(3, 0.0);
        std::vector<double> aligned_tangent(3, 0.0);

        aligner.align_orbit_closest_to_reference(
            reference,
            translated,
            aligned);
        require(distance(aligned, reference) < 1.0e-14,
                "residual translation aligns the state");
        aligner.apply_last_alignment(
            translated_tangent,
            aligned_tangent);
        require(distance(aligned_tangent, expected_tangent) < 1.0e-14,
                "the selected residual character also aligns the tangent");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
