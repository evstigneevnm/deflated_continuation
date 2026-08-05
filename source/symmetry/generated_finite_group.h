#ifndef __SYMMETRY_GENERATED_FINITE_GROUP_H__
#define __SYMMETRY_GENERATED_FINITE_GROUP_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace symmetry
{

template<class Element>
class generated_finite_group
{
public:
    using element_type = Element;

    struct entry_type
    {
        std::string name;
        element_type element;
        std::size_t inverse_index = 0;
    };

    explicit generated_finite_group(
        element_type identity,
        const std::size_t maximum_order = 4096):
        identity_(std::move(identity)),
        maximum_order_(maximum_order)
    {
        if(maximum_order_ == 0)
        {
            throw std::invalid_argument(
                "generated_finite_group maximum order must be positive");
        }
    }

    void add_generator(std::string name, element_type generator)
    {
        require_not_finalized();
        if(name.empty())
        {
            throw std::invalid_argument(
                "generated_finite_group generator name must not be empty");
        }
        generators_.push_back(
            named_generator{std::move(name), std::move(generator)});
    }

    void finalize()
    {
        if(finalized_)
        {
            return;
        }

        entries_.clear();
        index_by_key_.clear();
        add_element("identity", identity_);

        std::vector<named_generator> operations;
        for(const auto& generator: generators_)
        {
            append_operation_if_new(operations, generator);
            const element_type inverse = generator.element.inverse();
            if(inverse != generator.element)
            {
                append_operation_if_new(
                    operations,
                    named_generator{
                        generator.name + "^-1",
                        inverse});
            }
        }

        for(std::size_t current = 0; current < entries_.size(); ++current)
        {
            for(const auto& operation: operations)
            {
                const element_type candidate =
                    entries_[current].element.compose(operation.element);
                const std::string candidate_name =
                    entries_[current].name == "identity"
                        ? operation.name
                        : entries_[current].name + "*" + operation.name;
                add_element(candidate_name, candidate);
            }
        }

        validate_closure();
        for(auto& entry: entries_)
        {
            entry.inverse_index =
                find_index_unchecked(entry.element.inverse());
        }
        fingerprint_ = calculate_fingerprint();
        finalized_ = true;
    }

    bool finalized() const
    {
        return finalized_;
    }

    std::size_t size() const
    {
        require_finalized();
        return entries_.size();
    }

    const entry_type& entry(const std::size_t index) const
    {
        require_finalized();
        if(index >= entries_.size())
        {
            throw std::out_of_range(
                "generated_finite_group element index is out of range");
        }
        return entries_[index];
    }

    const std::vector<entry_type>& entries() const
    {
        require_finalized();
        return entries_;
    }

    std::size_t find_index(const element_type& element) const
    {
        require_finalized();
        return find_index_unchecked(element);
    }

private:
    std::size_t find_index_unchecked(const element_type& element) const
    {
        const auto found = index_by_key_.find(element.key());
        if(found == index_by_key_.end())
        {
            throw std::logic_error(
                "generated_finite_group element is outside the generated closure");
        }
        return found->second;
    }

public:
    const std::string& fingerprint() const
    {
        require_finalized();
        return fingerprint_;
    }

private:
    struct named_generator
    {
        std::string name;
        element_type element;
    };

    void require_not_finalized() const
    {
        if(finalized_)
        {
            throw std::logic_error(
                "generated_finite_group cannot be modified after finalization");
        }
    }

    void require_finalized() const
    {
        if(!finalized_)
        {
            throw std::logic_error(
                "generated_finite_group must be finalized before use");
        }
    }

    void append_operation_if_new(
        std::vector<named_generator>& operations,
        const named_generator& operation) const
    {
        const std::string key = operation.element.key();
        const auto found = std::find_if(
            operations.begin(),
            operations.end(),
            [&key](const named_generator& existing)
            {
                return existing.element.key() == key;
            });
        if(found == operations.end() && operation.element != identity_)
        {
            operations.push_back(operation);
        }
    }

    void add_element(const std::string& name, const element_type& element)
    {
        const std::string key = element.key();
        if(index_by_key_.find(key) != index_by_key_.end())
        {
            return;
        }
        if(entries_.size() >= maximum_order_)
        {
            throw std::runtime_error(
                "generated_finite_group exceeded its maximum order; "
                "the generators may define an infinite group");
        }
        const std::size_t index = entries_.size();
        entries_.push_back(entry_type{name, element, 0});
        index_by_key_.emplace(key, index);
    }

    void validate_closure() const
    {
        for(const auto& left: entries_)
        {
            for(const auto& right: entries_)
            {
                const auto product = left.element.compose(right.element);
                if(index_by_key_.find(product.key()) == index_by_key_.end())
                {
                    throw std::logic_error(
                        "generated_finite_group closure validation failed");
                }
            }
        }
    }

    std::string calculate_fingerprint() const
    {
        std::vector<std::string> keys;
        keys.reserve(entries_.size());
        for(const auto& entry: entries_)
        {
            keys.push_back(entry.element.key());
        }
        std::sort(keys.begin(), keys.end());

        std::uint64_t hash = 1469598103934665603ULL;
        for(const auto& key: keys)
        {
            for(const unsigned char value: key)
            {
                hash ^= static_cast<std::uint64_t>(value);
                hash *= 1099511628211ULL;
            }
            hash ^= 0xffULL;
            hash *= 1099511628211ULL;
        }

        std::ostringstream result;
        result << std::hex << std::setw(16) << std::setfill('0') << hash;
        return result.str();
    }

    element_type identity_;
    std::size_t maximum_order_;
    std::vector<named_generator> generators_;
    std::vector<entry_type> entries_;
    std::unordered_map<std::string, std::size_t> index_by_key_;
    std::string fingerprint_;
    bool finalized_ = false;
};

} // namespace symmetry

#endif
