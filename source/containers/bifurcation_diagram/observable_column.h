#ifndef BIFURCATION_DIAGRAM_OBSERVABLE_COLUMN_H
#define BIFURCATION_DIAGRAM_OBSERVABLE_COLUMN_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace container
{
// Resolve labels once. Older problems without labels can supply an explicit index.
template <class T>
class observable_column
{
public:
    explicit observable_column( std::size_t index ) : index_( index ) {}
    template <class Problem>
    observable_column( const Problem &problem, const std::string &label )
    {
        const auto labels = problem.norm_bifurcation_diagram_labels();
        const auto it = std::find( labels.begin(), labels.end(), label );
        if ( it == labels.end() || std::find( it + 1, labels.end(), label ) != labels.end() )
            throw std::invalid_argument( "Missing or ambiguous diagram column: " + label );
        index_ = static_cast<std::size_t>( it - labels.begin() );
        values_.reserve( labels.size() );
    }
    template <class Problem, class State>
    T evaluate( Problem &problem, const State &physical_state )
    {
        values_.clear(); // Some historical callbacks append rather than replace.
        problem.norm_bifurcation_diagram( physical_state, values_ );
        return values_.at( index_ );
    }
private:
    std::size_t index_;
    std::vector<T> values_;
};
}
#endif
