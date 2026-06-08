#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include <symmetry/stabilized_storage.h>

namespace
{

int checks = 0;
int failures = 0;

void require(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        std::cout << "FAIL " << label << std::endl;
        ++failures;
    }
}

void require_size(const std::string& label, std::size_t value, std::size_t expected)
{
    ++checks;
    if(value != expected)
    {
        std::cout << "FAIL " << label << " value=" << value
                  << " expected=" << expected << std::endl;
        ++failures;
    }
}

} // namespace

int main()
{
    {
        using state_type = std::vector<double>;
        symmetry::stabilized_storage<state_type> storage(1e-6);

        require(storage.empty(), "initially empty");
        require(storage.add_if_new({1.0, 2.0, 3.0}), "add first real state");
        require(!storage.add_if_new({1.0 + 1e-8, 2.0, 3.0}), "reject near duplicate");
        require(storage.add_if_new({1.0, 2.1, 3.0}), "add distinct real state");
        require_size("real storage size", storage.size(), 2);

        const auto nearest = storage.nearest_distance({1.0, 2.0, 3.1});
        require(nearest.second == 0, "nearest index");
        require(nearest.first > 0.0, "nearest positive distance");
        storage.clear();
        require(storage.empty(), "empty after clear");
    }

    {
        using complex_type = std::complex<double>;
        using state_type = std::vector<complex_type>;
        symmetry::stabilized_storage<state_type> storage(1e-5);

        require(storage.add_if_new({complex_type(1.0, 2.0), complex_type(3.0, -1.0)}), "add first complex state");
        require(!storage.add_if_new({complex_type(1.0 + 1e-7, 2.0), complex_type(3.0, -1.0)}), "reject complex near duplicate");
        require(storage.add_if_new({complex_type(1.0, 2.0), complex_type(3.0, -1.01)}), "add complex distinct state");
        require_size("complex storage size", storage.size(), 2);
    }

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
