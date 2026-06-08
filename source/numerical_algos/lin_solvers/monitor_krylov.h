// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_MONITOR_KRYLOV_H__
#define __SCFD_MONITOR_KRYLOV_H__

#ifdef SCFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include "default_monitor.h"

namespace numerical_algos
{
namespace lin_solvers 
{

template<class VectorOperations, class Log>
class monitor_krylov: public default_monitor<VectorOperations,Log>
{
private:
    using parent_t = default_monitor<VectorOperations, Log>;
    using T = typename VectorOperations::scalar_type;
protected:
    using parent_t::convergence_history_;
public:    
    using parent_t::iters_performed;
    using log_type = typename parent_t::logged_obj_type;
    using parent_t::resid_norm_out;
    using parent_t::converged;
    using parent_t::tol_out;

    struct params : public parent_t::params
    {
        params(const std::string &log_prefix = "") : parent_t::params(log_prefix, "monitor_krylov::")
        {
        }
        #ifdef SCFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            parent_t::params::from_json(j);
        }
        nlohmann::json to_json() const
        {
            return parent_t::params::to_json();
        }
        #endif
    };

    monitor_krylov(const VectorOperations &vec_ops, 
                    Log *log = NULL, 
                    const params &prms = params() ):
    parent_t(vec_ops, log, prms)
    {}

    bool check_finished_by_ritz_estimate(const T& ritz_val)
    {
        parent_t::resid_norm_ = ritz_val;
        parent_t::check_valid_norm();
        log_type::info_f("iter = %d", iters_performed() );
        log_type::info_f("resid norm = %0.6e tol = %0.6e", resid_norm_out(), tol_out() );
        if (parent_t::prms_.save_convergence_history)
        {
            convergence_history_.emplace_back( iters_performed(), resid_norm_out() );
        }
        return converged();

    }


};



}
}



#endif