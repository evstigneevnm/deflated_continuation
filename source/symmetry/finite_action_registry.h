#ifndef __SYMMETRY_FINITE_ACTION_REGISTRY_H__
#define __SYMMETRY_FINITE_ACTION_REGISTRY_H__

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace symmetry
{

template<class VectorOperations>
class finite_action_registry
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using action_function = std::function<void(const vector_type&, vector_type&)>;

    struct action_entry
    {
        std::string name;
        action_function apply;
        action_function pullback;
    };

    explicit finite_action_registry(VectorOperations* vec_ops_):
        vec_ops(vec_ops_)
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument("finite_action_registry got null vector operations");
        }
        reset_to_identity();
    }

    VectorOperations* vector_operations() const
    {
        return vec_ops;
    }

    void clear()
    {
        actions.clear();
        definition_fingerprint_.clear();
        definition_fingerprint_explicit_ = false;
    }

    void reset_to_identity()
    {
        actions.clear();
        definition_fingerprint_ = "identity";
        definition_fingerprint_explicit_ = false;
        add(
            "identity",
            [this](const vector_type& source, vector_type& destination)
            {
                vec_ops->assign(source, destination);
            });
    }

    void add(std::string name, action_function apply, action_function pullback = action_function())
    {
        if(name.empty())
        {
            throw std::invalid_argument("finite_action_registry action name must not be empty");
        }
        if(!apply)
        {
            throw std::invalid_argument("finite_action_registry action has no apply function");
        }
        if(!pullback)
        {
            pullback = apply;
        }
        actions.push_back(action_entry{std::move(name), std::move(apply), std::move(pullback)});
    }

    std::size_t size() const
    {
        return actions.size();
    }

    void set_definition_fingerprint(std::string fingerprint)
    {
        if(fingerprint.empty())
        {
            throw std::invalid_argument(
                "finite_action_registry definition fingerprint must not be empty");
        }
        definition_fingerprint_ = std::move(fingerprint);
        definition_fingerprint_explicit_ = true;
    }

    const std::string& definition_fingerprint() const
    {
        return definition_fingerprint_;
    }

    bool has_explicit_definition_fingerprint() const
    {
        return definition_fingerprint_explicit_;
    }

    std::vector<std::string> action_names() const
    {
        std::vector<std::string> result;
        result.reserve(actions.size());
        for(const auto& action: actions)
        {
            result.push_back(action.name);
        }
        return result;
    }

    const std::string& name(const std::size_t index) const
    {
        return entry(index).name;
    }

    int find(const std::string& action_name) const
    {
        for(std::size_t i = 0; i < actions.size(); ++i)
        {
            if(actions[i].name == action_name)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void apply(const std::size_t index, const vector_type& source, vector_type& destination) const
    {
        entry(index).apply(source, destination);
    }

    void pullback(const std::size_t index, const vector_type& source, vector_type& destination) const
    {
        entry(index).pullback(source, destination);
    }

private:
    const action_entry& entry(const std::size_t index) const
    {
        if(index >= actions.size())
        {
            throw std::out_of_range("finite_action_registry action index is out of range");
        }
        return actions[index];
    }

private:
    VectorOperations* vec_ops;
    std::vector<action_entry> actions;
    std::string definition_fingerprint_ = "identity";
    bool definition_fingerprint_explicit_ = false;
};

} // namespace symmetry

#endif
