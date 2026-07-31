#ifndef __STABILITY_ANALYSIS_TRANSITION_STATE_ALIGNMENT_H__
#define __STABILITY_ANALYSIS_TRANSITION_STATE_ALIGNMENT_H__

#include <functional>
#include <stdexcept>

namespace stability
{
namespace analysis
{

template<class VectorOperations>
class transition_state_alignment
{
public:
    using vector_type = typename VectorOperations::vector_type;
    using function_type = std::function<
        void(
            const vector_type&,
            const vector_type&,
            vector_type&)>;

    explicit transition_state_alignment(
        VectorOperations* vector_operations)
        : vector_operations_(vector_operations)
    {
        if(vector_operations_ == nullptr)
            throw std::invalid_argument(
                "transition_state_alignment: vector operations are null");
        reset();
    }

    template<class Aligner>
    void set(Aligner* aligner)
    {
        if(aligner == nullptr)
            throw std::invalid_argument(
                "transition_state_alignment: aligner is null");
        align_ =
            [aligner](
                const vector_type& reference,
                const vector_type& source,
                vector_type& destination)
            {
                aligner->stabilize_closest_to_reference(
                    reference,
                    source,
                    destination);
            };
        identity_ = false;
    }

    void reset()
    {
        VectorOperations* const vector_operations =
            vector_operations_;
        align_ =
            [vector_operations](
                const vector_type&,
                const vector_type& source,
                vector_type& destination)
            {
                vector_operations->assign(source, destination);
            };
        identity_ = true;
    }

    void align(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination) const
    {
        align_(reference, source, destination);
    }

    bool is_identity() const
    {
        return identity_;
    }

private:
    VectorOperations* vector_operations_;
    function_type align_;
    bool identity_ = true;
};

} // namespace analysis
} // namespace stability

#endif
