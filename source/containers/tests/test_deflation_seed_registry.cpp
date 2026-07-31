#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <containers/deflation_seed_registry.h>

namespace
{

void require(
    const bool condition,
    const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    const auto file_name =
        std::filesystem::temp_directory_path()/
        "deflated_continuation_seed_registry_test.json";
    std::error_code error;
    std::filesystem::remove(file_name, error);

    try
    {
        {
            container::deflation_seed_registry<double>
                registry(file_name.string());
            require(
                registry.next(24.0) == 0,
                "new knot starts at seed zero");
            registry.advance(24.0, 4);
            registry.advance(24.0, 2);
            require(
                registry.next(24.0) == 6,
                "same-knot seed advances monotonically");
            require(
                registry.next(25.0) == 0,
                "different knot has an independent seed sequence");
            registry.save();
        }

        container::deflation_seed_registry<double>
            loaded(file_name.string());
        require(
            loaded.next(24.0) == 6,
            "seed sequence survives restart");
        require(
            loaded.all().size() == 1,
            "registry stores one entry per knot");
    }
    catch(const std::exception& exception)
    {
        std::filesystem::remove(file_name, error);
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::filesystem::remove(file_name, error);
    std::cout << "PASSED\n";
    return 0;
}
