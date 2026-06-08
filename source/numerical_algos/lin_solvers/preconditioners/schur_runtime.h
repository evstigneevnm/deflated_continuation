#ifndef __SCFD_PRECONDITIONER_SCHUR_RUNTIME_HPP__
#define __SCFD_PRECONDITIONER_SCHUR_RUNTIME_HPP__

#include <vector>
#include <memory>
#ifdef SCFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <common/detail/vector_wrap.h>
#include <glued_matrix_operator.h>
#include "dummy.h"
#include "schur_bfbt.hpp"

namespace numerical_algos
{
namespace preconditioners 
{

template <class USolver, class PSolver, class Operations, class Log>
class schur_runtime : public scfd::utils::logged_obj_base<Log>
{
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using internal_vector_t = typename Operations::vector_type;
    using internal_matrix_t = typename Operations::matrix_type;
public:
    using operations_type = Operations;
    using scalar_type = typename Operations::scalar_type;
    using operator_type = scfd::linspace::glued_matrix_operator<Operations,2>;
    using vector_type = typename operator_type::vector_type;
    using vector_space_type = typename operator_type::vector_space_type;

    using schur_base_t = schur_base<USolver,PSolver,Operations,Log>;
    using schur_bfbt_t = schur_bfbt<USolver,PSolver,Operations,Log>;
    using dummy_t = dummy<vector_space_type,operator_type>;
    using preconditioner_interface_t = 
        preconditioner_interface<vector_space_type,operator_type>;
    
    using usolver_params_hierarchy_type = typename algo_params_hierarchy<USolver>::type;
    using psolver_params_hierarchy_type = typename algo_params_hierarchy<PSolver>::type;
    struct params_hierarchy
    {
        std::string type;
        typename dummy_t::params dummy;
        typename schur_bfbt_t::params schur_bfbt;

        usolver_params_hierarchy_type usolver;
        psolver_params_hierarchy_type psolver;

        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "schur_runtime::") : 
            type("dummy"),
            dummy(log_prefix,log_name),
            schur_bfbt(log_prefix,log_name),
            usolver(schur_bfbt.log_msg_prefix + "(USolv)"),
            psolver(schur_bfbt.log_msg_prefix + "(PSolv)")
        {
        }
        #ifdef SCFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            type = j.value("type", type);
            if (type == "dummy")
                dummy.from_json(j);
            else if (type == "schur_bfbt")
                schur_bfbt.from_json(j);
            else 
                throw std::runtime_error("schur_runtime::wrong type: " + type);
            if (type != "dummy")
            {
                usolver.from_json(j.at("usolver"));
                psolver.from_json(j.at("psolver"));
            }
        }
        nlohmann::json to_json() const
        {
            nlohmann::json j, j_type{{"type", type}};
            if (type == "dummy")
                j = dummy.to_json();
            else if (type == "schur_bfbt")
                j = schur_bfbt.to_json();
            else 
                throw std::runtime_error("schur_runtime::wrong type: " + type);
            j.insert(j_type.begin(), j_type.end());
            if (type != "dummy")
            {
                nlohmann::json  j_usolver = usolver.to_json(),
                                j_psolver = psolver.to_json();
                j["usolver"] = j_usolver;
                j["psolver"] = j_psolver;
            }
            return j;
        }
        #endif
        typename dummy_t::params_hierarchy dummy_hierarchy()const
        {
            return typename dummy_t::params_hierarchy(dummy);
        }
        typename schur_bfbt_t::params_hierarchy schur_bfbt_hierarchy()const
        {
            return typename schur_bfbt_t::params_hierarchy(schur_bfbt, usolver, psolver);
        }
    };
    using usolver_utils_hierarchy_type = typename algo_utils_hierarchy<USolver>::type;
    using psolver_utils_hierarchy_type = typename algo_utils_hierarchy<PSolver>::type;
    using utils_hierarchy = typename schur_base_t::utils_hierarchy;

    schur_runtime(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()
    )
    {
        if (prm.type == "dummy")
            prec_ = std::unique_ptr<preconditioner_interface_t>(new dummy_t(utils.vec_space,prm.dummy_hierarchy()));
        else if (prm.type == "schur_bfbt")
            prec_ = std::unique_ptr<preconditioner_interface_t>(new schur_bfbt_t(utils,prm.schur_bfbt_hierarchy()));
        else
            throw std::runtime_error("schur_runtime::wrong type: " + prm.type);
    }

    void set_operator(std::shared_ptr<const operator_type> K)
    {
        prec_->set_operator(K);
    }

    void apply(const vector_type &rhs, vector_type &x) const 
    {
        prec_->apply(rhs,x);
    }
    void apply(vector_type &x) const 
    {
        prec_->apply(x);
    }


private:
    std::unique_ptr<preconditioner_interface_t> prec_;

};


}  // preconditioners
}  // numerical_algos

#endif