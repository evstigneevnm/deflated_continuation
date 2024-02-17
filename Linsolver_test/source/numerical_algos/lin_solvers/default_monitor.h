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

#ifndef __SCFD_DEFAULT_MONITOR_H__
#define __SCFD_DEFAULT_MONITOR_H__

#include <cmath>
#include <vector>
#include <utility>
#ifdef SCFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <scfd/utils/logged_obj_base.h>
#include <scfd/utils/log.h>
#include "detail/vectors_arr_wrap_static.h"

namespace numerical_algos
{
namespace lin_solvers 
{

template<class VectorOperations, class Log>
class default_monitor : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using vector_operations_type = VectorOperations;
    using logged_obj_type = scfd::utils::logged_obj_base<Log>;

private:
    using T = scalar_type;
    using buf_arr_t = detail::vectors_arr_wrap_static<VectorOperations,1>;
    const vector_operations_type &vec_ops_;

public:
    struct params : public logged_obj_type::params
    {
        T rel_tol;
        T abs_tol;
        int max_iters_num;
        int min_iters_num;
        bool out_min_resid_norm;
        bool save_convergence_history;
        bool divide_out_norms_by_rel_base;

        params(
            const std::string &log_pefix = "", const std::string &log_name = "default_monitor::"
        ) :
            logged_obj_type::params(0, log_pefix + log_name),
            rel_tol(1.0e-6), abs_tol(1.0e-15),max_iters_num(100),min_iters_num(0),
            out_min_resid_norm(false),save_convergence_history(false),
            divide_out_norms_by_rel_base(true)
        {
        }

        #ifdef SCFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            rel_tol = j.value("rel_tol", rel_tol);
            abs_tol = j.value("abs_tol", abs_tol);
            max_iters_num = j.value("max_iters_num", max_iters_num);
            min_iters_num = j.value("min_iters_num", min_iters_num);
            out_min_resid_norm = j.value("out_min_resid_norm", out_min_resid_norm);
            save_convergence_history = j.value("save_convergence_history", save_convergence_history);
            divide_out_norms_by_rel_base = j.value("divide_out_norms_by_rel_base", divide_out_norms_by_rel_base);
        }
        nlohmann::json to_json() const
        {
            return
                nlohmann::json
                {
                    {"rel_tol", rel_tol},
                    {"abs_tol", abs_tol},
                    {"max_iters_num", max_iters_num},
                    {"min_iters_num", min_iters_num},
                    {"out_min_resid_norm", out_min_resid_norm},
                    {"save_convergence_history", save_convergence_history},
                    {"divide_out_norms_by_rel_base", divide_out_norms_by_rel_base}
                };
        }
        #endif

    };

private:

    T rel_tol_save_;
    int max_iters_num_save_;
    
    //followings are current convergence info
    int iters_performed_;
    //is_valid_number is a flag, meaning whether current solution is a valid 
    //vector (without nans or infs)
    mutable bool is_valid_number_;
    T rhs_norm_;
    T min_resid_norm_;
    buf_arr_t buf_;
    vector_type &min_resid_norm_x_;


protected:
    std::vector< std::pair<int,T> >  convergence_history_;
    params prms_;
    T resid_norm_;

public:
    default_monitor(const vector_operations_type &vec_ops, 
                    Log *log = NULL, const params &prms = params() ): 
        logged_obj_type(log, prms),
        vec_ops_(vec_ops), buf_(&vec_ops),
        min_resid_norm_x_(buf_[0]), prms_(prms)
    {
       
        max_iters_num_save_ = prms_.max_iters_num;
        if (prms_.out_min_resid_norm) 
        {
            buf_.init();
        }
        rel_tol_save_ = prms_.rel_tol;
    }

    void init(T rel_tol, T abs_tol = T(0.f),
            int max_iters_num = 100, int min_iters_num = 0, 
            bool out_min_resid_norm = false,
            bool save_convergence_history = false,
            bool divide_out_norms_by_rel_base = false)
    {
        prms_.rel_tol = rel_tol; prms_.abs_tol = abs_tol;
        prms_.max_iters_num = max_iters_num; 
        prms_.min_iters_num = min_iters_num;
        max_iters_num_save_ = max_iters_num;
        prms_.out_min_resid_norm = out_min_resid_norm;
        if (prms_.out_min_resid_norm) buf_.init();
        prms_.save_convergence_history = save_convergence_history;
        prms_.divide_out_norms_by_rel_base = divide_out_norms_by_rel_base;
        rel_tol_save_ = rel_tol;
    }


    void set_save_convergence_history(bool save_convergence_history)
    {
        prms_.save_convergence_history = save_convergence_history;
    }
    void set_divide_out_norms_by_rel_base(bool divide_out_norms_by_rel_base)
    {
        prms_.divide_out_norms_by_rel_base = divide_out_norms_by_rel_base;
    }
    //TODO init with config
    //TODO add separate function to control tolerances and behaviour
    void set_temp_tolerance(T rel_tol)
    {
        rel_tol_save_ = rel_tol;
        prms_.rel_tol = rel_tol;   
    }
    void restore_tolerance()
    {
        prms_.rel_tol = rel_tol_save_;
    }
    void set_temp_max_iterations(int max_iter_local)
    {
        max_iters_num_save_ = prms_.max_iters_num;
        prms_.max_iters_num = max_iter_local;
    }
    void restore_max_iterations()
    {
        prms_.max_iters_num = max_iters_num_save_;
    }


    void start(const vector_type& rhs)
    {
        rhs_norm_ = vec_ops_.norm(rhs);
        iters_performed_ = 0;
        if (prms_.out_min_resid_norm) buf_.start_use_all();
        if (prms_.save_convergence_history) convergence_history_.clear();
    }
    void stop()
    {
        if (prms_.out_min_resid_norm) buf_.stop_use_all();
    }

    T rel_tol()const { return prms_.rel_tol; }
    T abs_tol()const { return prms_.abs_tol; }
    T rel_tol_base()const { return rhs_norm(); }
    T tol()const 
    { 
        return abs_tol() + rel_tol()*rel_tol_base(); 
    }
    int max_iters_num()const
    {
        return prms_.max_iters_num; 
    } 
    int min_iters_num()const { return prms_.min_iters_num; }
    bool out_min_resid_norm()const { return prms_.out_min_resid_norm; }
    bool save_convergence_history()const { return prms_.save_convergence_history; }
    bool divide_out_norms_by_rel_base()const { return prms_.divide_out_norms_by_rel_base; }

    int iters_performed()const { return iters_performed_; }
    bool is_valid_number()const { return is_valid_number_; }
    void check_valid_norm()const
    {
        is_valid_number_ = std::isfinite(resid_norm_);
    }
    T rhs_norm()const { return rhs_norm_; }
    T resid_norm()const { return resid_norm_; }
    T resid_norm_out()const 
    { 
        if (!divide_out_norms_by_rel_base())
            return resid_norm(); 
        else
            return resid_norm()/rel_tol_base();
    }
    T tol_out()const 
    { 
        if (!divide_out_norms_by_rel_base())
            return tol(); 
        else
            return tol()/rel_tol_base();
    }
    const vector_type &min_resid_norm_x()const { return min_resid_norm_x_; }

    const std::vector<std::pair<int,T> > &convergence_history()const { return convergence_history_; }

    default_monitor &operator++() 
    {  
        ++iters_performed_; 
        return *this;
    }
    default_monitor &operator+=(int n) 
    {  
        iters_performed_ += n; 
        return *this;
    }
    bool converged()const
    {
        return is_valid_number() && (resid_norm() <= tol());
    }
    bool check_finished(const vector_type& x, const vector_type& r)
    {
        logged_obj_type::info_f("iter = %d, max_iters_num = %d", iters_performed(), max_iters_num() );

        is_valid_number_ = vec_ops_.is_valid_number(x);
        if (!is_valid_number_) 
        {
            logged_obj_type::info_f("solution is not a valid number");
            return true;
        }

        resid_norm_ = vec_ops_.norm(r);

        logged_obj_type::info_f("resid norm = %0.6e tol = %0.6e", resid_norm_out(), tol_out());
        if (prms_.save_convergence_history) 
            convergence_history_.emplace_back( iters_performed(), resid_norm_out() );

        if (out_min_resid_norm()) {
            if ((iters_performed() == 0)||(resid_norm() < min_resid_norm_)) {
                min_resid_norm_ = resid_norm();
                vec_ops_.assign(x, min_resid_norm_x_);
            }
        }
        
        return (converged() && iters_performed() >= min_iters_num()) || iters_performed() >= max_iters_num();
    }
};

}
}

#endif