#ifndef __BIFURCATION_DIAGRAM_CURVE_VECTOR_STORE_H__
#define __BIFURCATION_DIAGRAM_CURVE_VECTOR_STORE_H__

#include <cstdint>
#include <string>
#include <utility>

namespace container
{

template<class VectorFileOperations, class Log, class Vector>
class curve_vector_store
{
public:
    using store_result = std::pair<bool, uint64_t>;

    curve_vector_store() = default;

    curve_vector_store(VectorFileOperations* file_operations, Log* log):
        file_operations_(file_operations),
        log_(log)
    {
    }

    void bind(VectorFileOperations* file_operations, Log* log)
    {
        file_operations_ = file_operations;
        log_ = log;
    }

    store_result store(
        const std::string& directory,
        const unsigned int skip_output,
        uint64_t& global_index,
        uint64_t& global_id,
        const Vector& vector,
        const bool force_store)
    {
        const bool should_store = ((global_index++)%skip_output == 0) || force_store;
        if(!should_store)
        {
            return {false, 0};
        }

        const uint64_t id = ++global_id;
        const std::string file_name = directory + "/" + std::to_string(id);
        if(log_ != nullptr)
        {
            log_->info_f("container::bifurcation_diagram_curve: FULL PATH: %s", directory.c_str());
        }
        file_operations_->write_vector(file_name, vector);
        return {true, id};
    }

    void read(const std::string& directory, const uint64_t id, Vector& vector) const
    {
        file_operations_->read_vector(directory + "/" + std::to_string(id), vector);
    }

private:
    VectorFileOperations* file_operations_ = nullptr;
    Log* log_ = nullptr;
};

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_VECTOR_STORE_H__
