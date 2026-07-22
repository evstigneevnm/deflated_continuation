#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <containers/bifurcation_diagram/curve_vector_store.h>

namespace
{

struct fake_file_operations
{
    void write_vector(const std::string& file_name, const int& value)
    {
        data[file_name] = value;
    }

    void read_vector(const std::string& file_name, int& value)
    {
        value = data.at(file_name);
    }

    std::unordered_map<std::string, int> data;
};

struct fake_log
{
    void info_f(const char*, const char*)
    {
    }
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

}

int main()
{
    try
    {
        fake_file_operations files;
        fake_log log;
        container::curve_vector_store<fake_file_operations, fake_log, int> store(&files, &log);
        uint64_t index = 0;
        uint64_t id = 0;

        const auto first = store.store("curve", 3, index, id, 10, false);
        const auto second = store.store("curve", 3, index, id, 20, false);
        const auto third = store.store("curve", 3, index, id, 30, false);
        const auto fourth = store.store("curve", 3, index, id, 40, false);
        const auto forced = store.store("curve", 3, index, id, 50, true);

        require(first.first && first.second == 1, "first point storage");
        require(!second.first && !third.first, "skip cadence");
        require(fourth.first && fourth.second == 2, "periodic point storage");
        require(forced.first && forced.second == 3, "forced point storage");
        require(index == 5 && id == 3, "counter updates");

        int value = 0;
        store.read("curve", 1, value);
        require(value == 10, "first vector read");
        store.read("curve", 3, value);
        require(value == 50, "forced vector read");
    }
    catch(const std::exception& exception)
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
