/**
General solver for the block linear system of size (n+1 \times n+1) (rank one update):
    
    GAX=GB

where:
A=[LinearOperator d;c^T alpha]; 
X=[u v];  
B=[b beta];
G=f(Preconditioner) is the preconditioner build by the Sherman Morrison formula f(.).

input:
LinearOperator is a matrix-vector operator. In general it is a matrix of size nXn
d and c are contravariant vectors of size n;
b is an RHS contravariant vector of size n;
alpha and beta are scalars
Preconditioner is the preconditioner operator. In general it is a matrix such, that Preconditioner \seq inv(LinearOperator).

output:
u is a contravariant vector of size n;
v is a scalar

*/


#ifndef __SHERMAN_MORRISON_LINEAR_SYSTEM_SOLVE_H__
#define __SHERMAN_MORRISON_LINEAR_SYSTEM_SOLVE_H__

#include <numerical_algos/detail/vectors_arr_wrap_static.h>
#include <utils/logged_obj_base.h>

namespace numerical_algos
{
namespace sherman_morrison_linear_system
{

using numerical_algos::detail::vectors_arr_wrap_static;


template<class LinearOperator, class Preconditioner, class VectorOperations, class Monitor, class Log, template<class , class , class , class , class > class LinearSolver>
class sherman_morrison_linear_system_solve: public utils::logged_obj_base<Log>
{
public:
    typedef typename VectorOperations::scalar_type  scalar_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef LinearOperator                          linear_operator_type;
    typedef Preconditioner                          preconditioner_type;
    typedef VectorOperations                        vector_operations_type;
    typedef Monitor                                 monitor_type;
    typedef Log                                     log_type;

private:
    typedef vectors_arr_wrap_static<VectorOperations, 1> buf_t;
    typedef utils::logged_obj_base<Log>                  logged_obj_t;
    
    // let's use nested classes?!

    // modified Linear Operator for extended system
    struct LinearOperator_SM
    {
        mutable const vector_operations_type *vector_operations;
        mutable const linear_operator_type *inherited_linear_operator;
        mutable vector_type b, c, d;
        mutable scalar_type alpha, beta, gamma, v;

        mutable vector_type z;
        mutable bool small_alpha=false;
        mutable bool use_small_alpha = true;
        const scalar_type alpha_threshold_;

        LinearOperator_SM(const vector_operations_type *vector_operations_, const scalar_type alpha_threshold_p):
        alpha_threshold_(alpha_threshold_p)
        {
            vector_operations=vector_operations_;
            vector_operations->init_vectors(z); vector_operations->start_use_vectors(z);

        }

        ~LinearOperator_SM()
        {
            vector_operations->stop_use_vectors(z); vector_operations->free_vectors(z);
        }

        void set_inherited_linear_operator(const linear_operator_type *inherited_linear_operator_) const
        {
            inherited_linear_operator=inherited_linear_operator_;
        }
        const linear_operator_type* get_inherited_linear_operator() const
        {
            return inherited_linear_operator;
        }
        
        void set_vectors(const vector_type& c_, const vector_type& d_, const scalar_type& alpha_, const vector_type& b_, const scalar_type& beta_) const
        {
            b = b_;
            c = c_;
            d = d_;
            alpha = alpha_;
            beta = beta_;
            gamma = scalar_type(1);
            v=0.0;
            if((use_small_alpha)&&(alpha<alpha_threshold_))
            {
                std::cout << "\n==============SMALL ALPHA===========\n";
                small_alpha=true;
            }
        }
        void set_vectors(const scalar_type& gamma_, const vector_type& c_, const vector_type& d_, const scalar_type& alpha_) const
        {
            c = c_;
            d = d_;
            alpha = alpha_;
            gamma = gamma_;
            v=0.0;
            if((use_small_alpha)&&(alpha<alpha_threshold_))
            {
                std::cout << "\n==============SMALL ALPHA===========\n";
                small_alpha=true;
            }            
        }        
        void scale_solution(vector_type &u) const
        {
            if(small_alpha)
            {
                vector_operations->scale(alpha, u);
            }
        }

        bool is_small_alpha_used() const
        {
            return small_alpha;
        }

        scalar_type get_scalar(vector_type &u) const
        {
            scalar_type cTu_over_alpha=(vector_operations->scalar_prod(c, u));
            v=(beta-cTu_over_alpha)/alpha;
            return v;
        }

        const vector_type &get_rhs() const
        {
            //calc: z := mul_x*x + mul_y*y
            if(small_alpha)
            {
                vector_operations->assign_mul(alpha, b, -beta, d, z); 
            }
            else
            {
                scalar_type beta_over_alpha=beta/alpha;
                vector_operations->assign_mul(scalar_type(1.0), b, -beta_over_alpha, d, z);
            }
            return z;

        }

        void apply(const vector_type& u, vector_type& f) const
        {
                        
            if(small_alpha)
            {
                inherited_linear_operator->apply(u, f);
            }
            else
            {
                scalar_type cTu_over_alpha=(vector_operations->scalar_prod(c, u))/alpha;
                inherited_linear_operator->apply(u, f);
                //f=gamma*f-d/alpha*(cTu);
                //calc: y := mul_x*x + mul_y*y
                vector_operations->add_mul(-cTu_over_alpha, d, gamma, f);
            }
        }



    };
    
    // modified Preconditioner
    struct Preconditioner_SM
    {
        mutable const vector_operations_type *vector_operations;
        mutable const LinearOperator_SM *op;
        mutable const preconditioner_type *inherited_preconditioner;

        mutable vector_type c, d, b_;
        mutable vector_type g;
        mutable scalar_type alpha;
        mutable scalar_type beta_;
        mutable scalar_type gamma_;
        mutable bool small_alpha = false;
        mutable bool use_small_alpha = true;
        const scalar_type alpha_threshold_;

        Preconditioner_SM(const vector_operations_type *vector_operations_, const scalar_type alpha_threshold_p):
        alpha_threshold_(alpha_threshold_p)
        {
            vector_operations = vector_operations_;
            
            vector_operations->init_vector(g); vector_operations->start_use_vector(g);

        }
        ~Preconditioner_SM()
        {
            vector_operations->stop_use_vector(g); vector_operations->free_vector(g);            
        }

        void set_operator(const LinearOperator_SM *op_) const
        {
            op = op_;
            if(inherited_preconditioner != nullptr)
            {
                inherited_preconditioner->set_operator( op->get_inherited_linear_operator() ); 
            }
        }
        void set_vectors(const vector_type& c_, const vector_type& d_, const scalar_type& alpha_, const vector_type& b_p, const scalar_type& beta_p) const
        {
            b_ = b_p;
            c = c_;
            d = d_;
            beta_ = beta_p;
            alpha = alpha_;
            gamma_ = scalar_type(1.0);
            if((alpha<alpha_threshold_)&&(use_small_alpha))
            {
                small_alpha=true;
            } 
        }
        void set_vectors(const scalar_type& gamma_p, const vector_type& c_, const vector_type& d_, const scalar_type& alpha_, const vector_type& b_p) const
        {
            b_ = b_p;
            c = c_;
            d = d_;
            beta_ = 0;
            alpha = alpha_;
            gamma_ = gamma_p;
            if((alpha<alpha_threshold_)&&(use_small_alpha))
            {
                small_alpha=true;
            } 
        }        
        void set_inherited_preconditioner(const preconditioner_type *inherited_preconditioner_) const
        {
            inherited_preconditioner = inherited_preconditioner_;

        }
        void apply(vector_type& x)const
        {
            if(small_alpha)
            {
                inherited_preconditioner->apply(x);
            }
            else
            {
                //calc: y := mul_x*x
                vector_operations->assign_mul(-1.0, d, g); // g=-1/alpha*d
                inherited_preconditioner->apply(x); // inv(A)u=>x
                inherited_preconditioner->apply(g); // inv(A)g=>g
                
                scalar_type dot_prod_num = vector_operations->scalar_prod(c, x)/gamma_; // (c,Au):=(c,x)=>cx
                scalar_type dot_prod_din = vector_operations->scalar_prod(c, g)/gamma_; // (c,Ag):=(c,g)=>cg
                scalar_type mul_g=-dot_prod_num/(alpha+dot_prod_din);
                // ( x/beta - g/beta*(cx)/(beta+cg) ) -> x

                //calc: y := mul_x*x + mul_y*y
                vector_operations->add_mul(mul_g/gamma_, g, scalar_type(1.0)/gamma_, x);

            }
        }

    };

public:
    typedef LinearOperator_SM LinearOperator_SM_t;
    typedef Preconditioner_SM Preconditioner_SM_t;

    typedef LinearSolver<LinearOperator_SM_t, 
                        Preconditioner_SM_t,
                        VectorOperations,
                        Monitor, Log>           linear_solver_type;

    typedef LinearSolver<LinearOperator, 
                        Preconditioner,
                        VectorOperations,
                        Monitor, Log>           linear_solver_original_type;

    sherman_morrison_linear_system_solve(Preconditioner* prec_, const VectorOperations* vec_ops_p, Log *log_p = nullptr, int obj_log_lev = 2): 
    linear_solver(vec_ops_p, log_p), 
    prec(vec_ops_p, alpha_threshold_), 
    oper(vec_ops_p, alpha_threshold_),
    linear_solver_original(vec_ops_p, log_p),
    vec_ops_(vec_ops_p),
    utils::logged_obj_base<Log>(log_p, obj_log_lev, "sherman_morrison:") 
    {
        
        linear_solver.set_preconditioner(&prec);
        prec.set_inherited_preconditioner(prec_);
        //sets preconditioner for the original linear solver
        linear_solver_original.set_preconditioner(prec_);
        //init_residual_vector
        vec_ops_->init_vector(r_);
        vec_ops_->start_use_vector(r_);

    }

    ~sherman_morrison_linear_system_solve()
    {
        vec_ops_->stop_use_vector(r_);
        vec_ops_->free_vector(r_);
    }


    linear_solver_type *get_linsolver_handle()
    {
        return &linear_solver;
    }
    linear_solver_original_type *get_linsolver_handle_original()
    {
        return &linear_solver_original;
    }
    void is_small_alpha(bool use_small_alpha_)
    {
        oper.use_small_alpha = use_small_alpha_;
        prec.use_small_alpha = use_small_alpha_;
        small_alpha = use_small_alpha_;
    }

    //solves full extended system with rank 1 update
    bool solve(const LinearOperator &A, const vector_type &c, const vector_type &d, const scalar_type &alpha, const vector_type &b, const scalar_type &beta, vector_type &u, scalar_type &v) const
    {
        bool flag = false;
        oper.set_inherited_linear_operator(&A);
        oper.set_vectors(c, d, alpha, b, beta);
        prec.set_vectors(c, d, alpha, b, beta);

        if( oper.is_small_alpha_used() )
        {
            bool flag1 = linear_solver.solve(oper, b, u);
            bool flag2 = linear_solver.solve(oper, d, r_);
            scalar_type aa = vec_ops_->scalar_prod(c, r_);
            scalar_type ba = vec_ops_->scalar_prod(c, u);
            v = (beta-ba)/(-aa+alpha);
            // std::cout << "(-aa+alpha*gamma) = " << (-aa+alpha*gamma) << " v = " << v << std::endl;
            vec_ops_->add_mul(-v, r_, 1.0, u);
            flag = flag2&flag1;
        }
        else
        {
            flag = linear_solver.solve(oper, oper.get_rhs(), u);
            v = oper.get_scalar(u);
        }
        vec_ops_->assign_scalar(0.0, r_);
        A.apply(u, r_);
        vec_ops_->add_mul(-1.0, b, v, d, 1.0, r_);
        scalar_type cTu = vec_ops_->scalar_prod(c, u);
        scalar_type r_sc_ = -beta+alpha*v+cTu;
        scalar_type residual_norm = vec_ops_->norm_rank1(r_, r_sc_);
        // std::cout << "actual residual = " << residual_norm << std::endl;
        logged_obj_t::info_f("actual residual = %e, alpha = %e, beta = %e", residual_norm, alpha, beta );
        return flag;
    }

    //solves Sherman - Morrison version as: 
    //    (gamma A - 1/alpha d c^t)u=b
    bool solve(const scalar_type &gamma, const LinearOperator &A, const scalar_type &alpha, const vector_type &c, const vector_type &d,  const vector_type &b, vector_type &u) const
    {
        bool flag = false;
        oper.set_inherited_linear_operator(&A);
        oper.set_vectors(gamma, c, d, alpha);
        prec.set_vectors(gamma, c, d, alpha, b); 

        if( oper.is_small_alpha_used() )
        {
            bool flag1 = linear_solver.solve(oper, b, u);
            bool flag2 = linear_solver.solve(oper, d, r_);
            scalar_type aa = vec_ops_->scalar_prod(c, r_);
            scalar_type ba = vec_ops_->scalar_prod(c, u);
            scalar_type v = -ba/(-aa+alpha*gamma);
            // std::cout << "(-aa+alpha*gamma) = " << (-aa+alpha*gamma) << " v = " << v << std::endl;
            vec_ops_->add_mul(-v/gamma, r_, 1.0/gamma, u);  
            flag = flag2&flag1;
        }
        else
        {
            flag = linear_solver.solve(oper, b, u);
            oper.scale_solution(u);
        }    
        A.apply(u, r_);
        vec_ops_->add_mul(-1.0, b, gamma, r_);
        scalar_type cTu = vec_ops_->scalar_prod(c, u);
        vec_ops_->add_mul(-cTu, d, static_cast<scalar_type>(alpha), r_);
        scalar_type residual_norm = vec_ops_->norm(r_);
        logged_obj_t::info_f("actual residual = %e, gamma = %e, alpha = %e", residual_norm, gamma, alpha );

        return flag;
    }

    //solves the original system Au=b
    //this method simply passes all to the original linear solver used in the constructor
    //this avoids adding another linear solver object explicitely
    bool solve(const LinearOperator &A,  const vector_type &b, vector_type &u) const
    {
        bool flag = false;
        flag = linear_solver_original.solve(A, b, u);
        return flag;
    }


private:
    linear_solver_type linear_solver;
    linear_solver_original_type linear_solver_original;
    const scalar_type alpha_threshold_ = 1.0e1;

    LinearOperator_SM_t oper;
    Preconditioner_SM_t prec;
    mutable bool small_alpha = true;
    mutable vector_type r_;
    mutable const vector_operations_type* vec_ops_;
    mutable Log* log_;


};

}
}


#endif