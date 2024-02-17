#ifndef __SCFD_PRECONDITIONER_SCHUR_BFBT_HPP__
#define __SCFD_PRECONDITIONER_SCHUR_BFBT_HPP__

#include <vector>
#include <memory>
#ifdef SCFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <common/detail/vector_wrap.h>
#include <solver/detail/algo_utils_hierarchy.h>
#include <solver/detail/algo_params_hierarchy.h>
#include <solver/detail/algo_hierarchy_creator.h>
#include <glued_matrix_operator.h>
#include "schur_base.h"
#include <profiling.h>

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
class schur_bfbt : public schur_base<USolver, PSolver, Operations, Log>
{
    using parent_t = schur_base<USolver, PSolver, Operations, Log>;
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;
    using internal_vector_t = typename Operations::vector_type;
    using internal_matrix_t = typename Operations::matrix_type;
    //template<class Algo, class P>
    //using numerical_algos::detail::algo_params_hierarchy<Algo,P>;
public:
    using operations_type = Operations;
    using scalar_type = typename Operations::scalar_type;
    using operator_type = scfd::linspace::glued_matrix_operator<Operations,2>;
    using vector_type = typename operator_type::vector_type;
    using vector_space_type = typename operator_type::vector_space_type;
    
    struct params : public logged_obj_params_t  
    {
        bool out_debug_matrices;
        bool use_scaling;

        params(const std::string &log_prefix = "", const std::string &log_name = "schur_bfbt::") :
            logged_obj_params_t(0, log_prefix+log_name),
            out_debug_matrices(false),
            use_scaling(false) 
        {
        }
        #ifdef SCFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            out_debug_matrices = j.value("out_debug_matrices", out_debug_matrices);
            use_scaling = j.value("use_scaling", use_scaling);
        }
        nlohmann::json to_json() const
        {
            return
                nlohmann::json
                {
                    {"type", "schur_bfbt"},
                    {"out_debug_matrices", out_debug_matrices},
                    {"use_scaling", use_scaling}
                };
        }
        #endif
    };
    using typename parent_t::utils;

    using typename parent_t::usolver_params_hierarchy_type;
    using typename parent_t::psolver_params_hierarchy_type;
    struct params_hierarchy : public params
    {
        usolver_params_hierarchy_type usolver;
        psolver_params_hierarchy_type psolver;

        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "schur_bfbt::") : 
            params(log_prefix, log_name),
            usolver(this->log_msg_prefix + "(USolv)"),
            psolver(this->log_msg_prefix + "(PSolv)")
        {
        }
        params_hierarchy(
            const params &prm_, 
            const usolver_params_hierarchy_type &usolver_,
            const psolver_params_hierarchy_type &psolver_
        ) : params(prm_), usolver(usolver_), psolver(psolver_)
        {
        }
        #ifdef SCFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            params::from_json(j);
            usolver.from_json(j.at("usolver"));
            psolver.from_json(j.at("psolver"));
        }
        nlohmann::json to_json() const
        {
            nlohmann::json  j = params::to_json(),
                            j_usolver = usolver.to_json(),
                            j_psolver = psolver.to_json();
            j["usolver"] = j_usolver;
            j["psolver"] = j_psolver;
            return j;
        }
        #endif
    };
    using typename parent_t::utils_hierarchy;

    schur_bfbt(
        std::shared_ptr<USolver> U, 
        std::shared_ptr<PSolver> P, 
        std::shared_ptr<operations_type> ops, 
        Log *log = nullptr,
        const params &prm = params()
    ) : 
        parent_t(U, P, ops, log, prm), prm_(prm)
    {
    }
    schur_bfbt(
        std::shared_ptr<const operator_type> K,
        std::shared_ptr<USolver> U, 
        std::shared_ptr<PSolver> P, 
        std::shared_ptr<operations_type> ops, 
        Log *log = nullptr,
        const params &prm = params()
    ) : schur_bfbt(U, P, ops, log, prm)
    {
        set_operator(std::move(K));
    }
    schur_bfbt(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()
    ) : 
        schur_bfbt(
            algo_hierarchy_creator<USolver>::get(utils.usolver,prm.usolver),
            algo_hierarchy_creator<PSolver>::get(utils.psolver,prm.psolver),
            utils.ops, utils.log, prm
        )
    {
    }

    void set_operator(std::shared_ptr<const operator_type> K)
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("schur_bfbt::set_operator");

        K_ = std::move(K);
        internal_bufs_ = std::make_shared<internal_bufs_t>(K_->get_im_space());

        if(prm_.use_scaling)
        {
            Mu = ops_->matrix_diag(A(), true);
        }

        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::set_operator:form_pressure_matrix");
        form_pressure_matrix();
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::set_operator:form_pressure_matrix");

        //TODO use shared_ptr
        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::set_operator:setup U solver");
        U_->set_operator(A_ptr());
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::set_operator:setup U solver");
        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::set_operator:setup P solver");
        P_->set_operator(Kp_schur);
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::set_operator:setup P solver");
    }


    void form_pressure_matrix()
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("schur_bfbt::form_pressure_matrix");

        if (prm_.out_debug_matrices) 
        {
            logged_obj_t::info("output debug matrices to files...");

            ops_->write_matrix_to_mm_file("cut_B.mtx", B());
            ops_->write_matrix_to_mm_file("cut_BT.mtx", BT());
            if (prm_.use_scaling)
            {
                ops_->write_matrix_to_mm_file("Mu.mtx", *Mu);
            }
            ops_->write_matrix_to_mm_file("cut_C.mtx", C());
        }        

        if (prm_.use_scaling)
        {
            auto Kp_schur_pre = ops_->matrix_matrix_prod(B(), *(ops_->matrix_matrix_prod(*Mu, BT())) );
            Kp_schur = ops_->matrix_matrix_sum(scalar_type(1), *Kp_schur_pre, scalar_type(1), C());
        }
        else
        {
            auto Kp_schur_pre = ops_->matrix_matrix_prod(B(), BT());
            Kp_schur = ops_->matrix_matrix_sum(scalar_type(1), *Kp_schur_pre, scalar_type(1), C());
        }

        if (prm_.out_debug_matrices) 
        {
            ops_->write_matrix_to_mm_file("cut_Kp_schur.mtx", *Kp_schur);
        }        
    }


    void apply(const vector_type &rhs, vector_type &x) const 
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("schur_bfbt::apply(rhs,x)");

        const internal_vector_t &rhs_u =  rhs.comp(0);
        const internal_vector_t &rhs_p =  rhs.comp(1);
        internal_vector_t &u =  x.comp(0);
        internal_vector_t &p =  x.comp(1);
        
        // S = B*Bt;
        // .
        // a = S\x_p;
        // b = (B*(A*(Bt*a)));
        // p = S\b;
        // %ends 
        // x_u = x_u - Bt*p;
        // u = A\x_u;  

        ops_->assign_scalar(scalar_type(0), p);
        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::apply:P solve");
        report("P1", P_->solve(rhs_p, p));
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::apply:P solve");

        //TODO change all add_ for assign_ (and add new asiign method in ops)

        ops_->add_matrix_vector_prod(1, BT(), p, 0, u);
        if(prm_.use_scaling)
        {
            ops_->add_matrix_vector_prod(1, *Mu, u, 0, a());
            ops_->add_matrix_vector_prod(1, A(), a(), 0, u);
            ops_->add_matrix_vector_prod(1, *Mu, u, 0, a());
        }
        else
        {
            ops_->add_matrix_vector_prod(1, A(), u, 0, a());
        }
        ops_->add_matrix_vector_prod(1, B(), a(), 0, b());

        ops_->assign_scalar(scalar_type(0), p);
        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::apply:P solve");
        report("P2", P_->solve( b(), p));
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::apply:P solve");

        ops_->assign(rhs_u, rhs_u_tmp());
        ops_->add_matrix_vector_prod(-1, BT(), p, 1, rhs_u_tmp());

        ops_->assign_scalar(scalar_type(0), u);
        STOKES_PORUS_3D_PLATFORM_TIC("schur_bfbt::apply:U solve");
        report("U3", U_->solve(rhs_u_tmp(), u));
        STOKES_PORUS_3D_PLATFORM_TOC("schur_bfbt::apply:U solve");
    }

    /// inplace version for preconditioner interface
    void apply(vector_type &x) const 
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("schur_bfbt::apply(x)");
        //TODO make start/stop use here for rhs_tmp()
        internal_bufs_->vec_space_->assign(x, rhs_tmp());
        apply(rhs_tmp(), x);
    }

protected:
    using parent_t::U_;
    using parent_t::P_;
    using parent_t::ops_;

private:
    using internal_space_t = typename vector_space_type::space_comp_type;
    using vector_wrap_t = scfd::linspace::detail::vector_wrap<vector_space_type,true,true>;
    using internal_vector_wrap_t = scfd::linspace::detail::vector_wrap<internal_space_t,true,true>;
    struct internal_bufs_t
    {
        std::shared_ptr<vector_space_type> vec_space_;
        internal_vector_wrap_t rhs_u_tmp_, a_, b_;
        vector_wrap_t rhs_tmp_;
        internal_bufs_t(std::shared_ptr<vector_space_type> vec_space) : 
            vec_space_(std::move(vec_space)), 
            rhs_u_tmp_(u_space()), a_(u_space()), b_(p_space()),
            rhs_tmp_(*vec_space_)
        {
        }
        internal_space_t &u_space()
        {
            return vec_space_->space_comp(0);
        }
        internal_space_t &p_space()
        {
            return vec_space_->space_comp(1);
        }
    };

    params prm_;

    std::shared_ptr<const operator_type> K_;

    std::shared_ptr<internal_matrix_t> Kp_schur;
    std::shared_ptr<internal_matrix_t> Mu;
    std::shared_ptr<internal_bufs_t> internal_bufs_;

    const internal_matrix_t &BT()const
    {
        return K_->comp(0,1);
    }
    const internal_matrix_t &B()const
    {
        return K_->comp(1,0);
    }
    const internal_matrix_t &A()const
    {
        return K_->comp(0,0);
    }
    const std::shared_ptr<internal_matrix_t> &A_ptr()const
    {
        return K_->comp_ptr(0,0);
    }
    const internal_matrix_t &C()const
    {
        return K_->comp(1,1);
    }

    internal_vector_t &rhs_u_tmp()const
    {
        return *(internal_bufs_->rhs_u_tmp_);
    }
    internal_vector_t &a()const
    {
        return *(internal_bufs_->a_);
    }
    internal_vector_t &b()const
    {
        return *(internal_bufs_->b_);
    }
    vector_type &rhs_tmp()const
    {
        return *(internal_bufs_->rhs_tmp_);
    }

    void report(const std::string &name, bool res) const 
    {
        //TODO temporaly set 2 to shut it up
        //instead it is better to add logged_obj_base::params reading from json
        //and use obj_log_lev parameter in config for fine tuning of per objects verbosity
        logged_obj_t::info_f(2, "%s: res = %d", name.c_str(), res);
    }


    /*friend std::ostream& operator<<(std::ostream &os, const schur_bfbt &p) {
        os << "Schur complement block triangular solver (two-stage preconditioner)" << std::endl;
        os << "  Unknowns: " << p.n << "(" << "u:" << p.nu << "p:" << p.np << ")" << std::endl;
        os << "  Memory:  " << human_readable_memory( p.bytes() + p.systemcut->bytes() ) << std::endl;
        os << std::endl;
        os << "[ U ]\n" << *p.U_ << std::endl;
        os << "[ P ]\n" << *p.P_ << std::endl;

        return os;
    }*/

};


}  // preconditioners
}  // numerical_algos

#endif