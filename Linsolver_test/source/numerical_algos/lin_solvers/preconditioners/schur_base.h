#ifndef __SCFD_PRECONDITIONER_SCHUR_BASE_H__
#define __SCFD_PRECONDITIONER_SCHUR_BASE_H__

#include <vector>
#include <memory>
#include <common/detail/vector_wrap.h>
#include <solver/detail/algo_utils_hierarchy.h>
#include <solver/detail/algo_params_hierarchy.h>
#include <solver/detail/algo_hierarchy_creator.h>
#include <glued_matrix_operator.h>
#include "preconditioner_interface.h"

namespace numerical_algos
{
namespace preconditioners 
{

/// block_triangular_bfbt (commutation) //block_triangular_simple (like SIMPLE)
/// braes_sarazin_bfbt//braes_sarazin_simple

template <class Operations>
using schur_operator = scfd::linspace::glued_matrix_operator<Operations,2>;
template <class Operations>
using schur_vector_space = typename schur_operator<Operations>::vector_space_type;

template <class USolver, class PSolver, class Operations, class Log>
class schur_base : 
    public preconditioner_interface
    <
        schur_vector_space<Operations>,schur_operator<Operations>
    >, 
    public scfd::utils::logged_obj_base<Log>
{
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;
    using internal_vector_t = typename Operations::vector_type;
    using internal_matrix_t = typename Operations::matrix_type;
public:
    using operations_type = Operations;
    using scalar_type = typename Operations::scalar_type;
    using operator_type = scfd::linspace::glued_matrix_operator<Operations,2>;
    using vector_type = typename operator_type::vector_type;
    using vector_space_type = typename operator_type::vector_space_type;
    
    struct utils
    { 
        std::shared_ptr<vector_space_type> vec_space;
        std::shared_ptr<operations_type> ops;
        Log *log;    

        utils() = default;
        utils(
            std::shared_ptr<vector_space_type> vec_space_,
            std::shared_ptr<operations_type> ops_, Log *log_ = nullptr
        ) : 
            vec_space(vec_space_), ops(ops_), log(log_)
        {
        }   
    };

    using usolver_params_hierarchy_type = typename algo_params_hierarchy<USolver>::type;
    using psolver_params_hierarchy_type = typename algo_params_hierarchy<PSolver>::type;
    using usolver_utils_hierarchy_type = typename algo_utils_hierarchy<USolver>::type;
    using psolver_utils_hierarchy_type = typename algo_utils_hierarchy<PSolver>::type;
    struct utils_hierarchy : public utils
    {
        usolver_utils_hierarchy_type usolver;
        psolver_utils_hierarchy_type psolver;

        utils_hierarchy() = default;
        template<class ...Args>
        utils_hierarchy(
            usolver_utils_hierarchy_type usolver_,
            psolver_utils_hierarchy_type psolver_,
            Args... args
        ) : 
            utils(args...),
            usolver(usolver_),
            psolver(psolver_)
        {        
        }
    };

    schur_base(
        std::shared_ptr<USolver> U, 
        std::shared_ptr<PSolver> P, 
        std::shared_ptr<operations_type> ops, 
        Log *log,
        const logged_obj_params_t &logged_obj_params
    ) : 
        logged_obj_t(log, logged_obj_params),
        U_(U), P_(P), ops_(ops)
    {
    }

protected:
    std::shared_ptr<USolver> U_;
    std::shared_ptr<PSolver> P_;
    std::shared_ptr<operations_type> ops_;
};


}  // preconditioners
}  // numerical_algos

#endif