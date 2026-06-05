
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <scfd/utils/log_std.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/jacobi.h>
#include <numerical_algos/lin_solvers/cgs.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <common/scfd_serial_cpu_vector_operations.h>

using namespace numerical_algos::lin_solvers;

typedef SCALAR_TYPE   real;
using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
using vector_t = typename vec_ops_t::vector_type;

//TODO now works only for a > 0
struct system_operator
{
    int     sz_;
    real    a_, re_, h_;

    system_operator(int sz, real a, real re) : sz_(sz), a_(a), re_(re), h_(real(1)/real(sz)) {}

    void apply(const vector_t& x, vector_t& f)const
    {
        for (int i = 0;i < sz_;++i) {
            real    xm = (i > 0     ? x(i-1) : real(0.f)),
                    xp = (i+1 < sz_ ? x(i+1) : real(0.f));
            f(i) = a_*(x(i)-xm)/h_ - (real(1.f)/re_)*(xp - real(2.f)*x(i) + xm)/(h_*h_);
            //f(i) = (real(1.f)/re_)*(xp - real(2.f)*x(i) + xm)/(h_*h_);
        }
    }
};

//TODO now works only for a > 0
struct prec_operator
{
    const system_operator     *op_;

    prec_operator() {}

    void set_operator(const system_operator *op)
    {
        op_ = op;
    }

    void apply(vector_t& x)const
    {
        for (int i = 0;i < op_->sz_;++i) {
            x(i) /= (op_->a_/op_->h_ - (real(1.f)/op_->re_)*(-real(2.f))/(op_->h_*op_->h_));

            //x(i) /= ((real(1.f)/re_)*(-real(2.f))/(h_*h_));
        }
    }
};

typedef scfd::utils::log_std                                                            log_t;
typedef default_monitor<vec_ops_t,log_t>                                                monitor_t;
typedef jacobi<system_operator,prec_operator,vec_ops_t,monitor_t,log_t>                 lin_solver_jacobi_t;
typedef cgs<system_operator,prec_operator,vec_ops_t,monitor_t,log_t>                    lin_solver_cgs_t;
typedef bicgstab<system_operator,prec_operator,vec_ops_t,monitor_t,log_t>               lin_solver_bicgstab_t;
typedef bicgstabl<system_operator,prec_operator,vec_ops_t,monitor_t,log_t>              lin_solver_bicgstabl_t;

void write_convergency(const std::string &fn, const std::vector<std::pair<int,real> > &conv, real tol)
{
    std::ofstream f(fn.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("write_convergency: error while opening file " + fn);

    for (int i = 0;i < conv.size();++i) {
        if (!(f << conv[i].first << " " << conv[i].second << " " << tol << std::endl)) 
            throw std::runtime_error("write_convergency: error while writing to file " + fn);
    }
}

int main(int argc, char **args)
{
    if (argc < 12) {
        std::cout << "USAGE: " << std::string(args[0]) << " <mesh_sz> <a> <re> <max_iters> <rel_tol> <use_precond_resid> <resid_recalc_freq> <basis_sz> <lin_solver_type> <result_fn> <convergency_fn>"  << std::endl;
        std::cout << "EXAMPLE: " << std::string(args[0]) << " 100 1. 100. 100 1e-7 1 0 2 0 test_out.dat conv_out.dat"  << std::endl;
        return 0;
    }

    std::string             res_fn(args[10]), conv_fn(args[11]);
    int                     sz = atoi(args[1]);
    real                    a = atof(args[2]),
                            re = atoi(args[3]);
    int                     max_iters = atoi(args[4]);
    real                    rel_tol = atof(args[5]);
    bool                    use_precond_resid = atoi(args[6]);
    int                     resid_recalc_freq = atoi(args[7]);
    int                     basis_sz = atoi(args[8]);
    int                     lin_solver_type = atoi(args[9]);

    log_t                   log;
    vec_ops_t               vec_ops(sz);
    system_operator         A(sz, a, re);
    vector_t                rhs, x;

    prec_operator           prec;
    lin_solver_jacobi_t     lin_solver_jacobi(&vec_ops, &log);
    lin_solver_cgs_t        lin_solver_cgs(&vec_ops, &log);
    lin_solver_bicgstab_t   lin_solver_bicgstab(&vec_ops, &log);
    lin_solver_bicgstabl_t  lin_solver_bicgstabl(&vec_ops, &log);
    monitor_t               *mon;

    switch (lin_solver_type) {
        case 0: mon = &lin_solver_jacobi.monitor(); break;
        case 1: mon = &lin_solver_cgs.monitor(); break;
        case 2: mon = &lin_solver_bicgstab.monitor(); break;
        case 3: mon = &lin_solver_bicgstabl.monitor(); break;
        default: throw std::runtime_error("unknown solvert type");
    }

    vec_ops.init_vector(x);
    vec_ops.init_vector(rhs);
    vec_ops.start_use_vector(x);
    vec_ops.start_use_vector(rhs);

    vec_ops.assign_scalar(real(1.f), rhs);
    vec_ops.assign_scalar(real(0.f), x);

    mon->init(rel_tol, real(0.f), max_iters);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    lin_solver_jacobi.set_preconditioner(&prec);
    lin_solver_cgs.set_preconditioner(&prec);
    lin_solver_bicgstab.set_preconditioner(&prec);
    lin_solver_bicgstabl.set_preconditioner(&prec);
    
    lin_solver_bicgstab.set_use_precond_resid(use_precond_resid);
    lin_solver_bicgstab.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver_bicgstabl.set_use_precond_resid(use_precond_resid);
    lin_solver_bicgstabl.set_resid_recalc_freq(resid_recalc_freq);

    lin_solver_bicgstabl.set_basis_size(basis_sz);

    bool    res_flag_;
    switch (lin_solver_type) {
        case 0: res_flag_ = lin_solver_jacobi.solve(A, rhs, x); break;
        case 1: res_flag_ = lin_solver_cgs.solve(A, rhs, x); break;
        case 2: res_flag_ = lin_solver_bicgstab.solve(A, rhs, x); break;
        case 3: res_flag_ = lin_solver_bicgstabl.solve(A, rhs, x); break;
        default: throw std::runtime_error("unknown solvert type");
    }
    int     iters_performed_ = mon->iters_performed();

    if (res_flag_) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");

    if (res_fn != "none") {
        std::ofstream    out_f(res_fn.c_str());
        for (int i = 0;i < sz;++i) {
            out_f << (i + real(0.5f))/sz << " " << x(i) << std::endl;
        }
        out_f.close();
    }
    if (conv_fn != "none") write_convergency(conv_fn, mon->convergence_history(), mon->tol_out());

    vec_ops.stop_use_vector(x);
    vec_ops.stop_use_vector(rhs);
    vec_ops.free_vector(x);
    vec_ops.free_vector(rhs);

    return 0;
}
