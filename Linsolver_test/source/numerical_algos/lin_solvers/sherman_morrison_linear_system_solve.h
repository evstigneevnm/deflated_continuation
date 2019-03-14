/*
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

namespace numerical_algos
{
namespace sherman_morrison_linear_system
{

using numerical_algos::detail::vectors_arr_wrap_static;


template<class LinearOperator, class Preconditioner, class VectorOperations, class Monitor, class Log, template<class , class , class , class , class > class LinearSolver>
class sherman_morrison_linear_system_solve
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

    // let's use nested classes?!

    // modified Linear Operator for extended system
    struct LinearOperator_SM
    {
        mutable const vector_operations_type *vector_operations;
        mutable const linear_operator_type *inherited_linear_operator;
        mutable vector_type b, c, d;
        mutable scalar_type alpha, beta, gamma, v;

        mutable vector_type z;

        LinearOperator_SM(const vector_operations_type *vector_operations_)
        {
            vector_operations=vector_operations_;
            vector_operations->init_vector(z); vector_operations->start_use_vector(z);

        }

        ~LinearOperator_SM()
        {
            vector_operations->stop_use_vector(z); vector_operations->free_vector(z);
        }

        void set_inherited_linear_operator(const linear_operator_type *inherited_linear_operator_) const
        {
            inherited_linear_operator=inherited_linear_operator_;
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
        }
        void set_vectors(const scalar_type& gamma_, const vector_type& c_, const vector_type& d_, const scalar_type& alpha_) const
        {
            c = c_;
            d = d_;
            alpha = alpha_;
            gamma = gamma_;
            v=0.0;
        }        
        scalar_type get_scalar(vector_type &u) const
        {
            scalar_type cTu_over_alpha=(vector_operations->scalar_prod(c, u))/alpha;
            v=beta/alpha-cTu_over_alpha;
            return v;
        }

        const vector_type &get_rhs() const
        {
            //calc: z := mul_x*x + mul_y*y
            scalar_type beta_over_alpha=beta/alpha;
            vector_operations->assign_mul(scalar_type(1.0), b, -beta_over_alpha, d, z);
            return z;
        }

        void apply(const vector_type& u, vector_type& f) const
        {
                        
            scalar_type cTu_over_alpha=(vector_operations->scalar_prod(c, u))/alpha;
            inherited_linear_operator->apply(u, f);
            //f=gamma*f-d/alpha*(cTu);
            //calc: y := mul_x*x + mul_y*y
            vector_operations->add_mul(-cTu_over_alpha, d, gamma, f);
        }



    };
    
    // modified Preconditioner
    struct Preconditioner_SM
    {
        mutable const vector_operations_type *vector_operations;
        mutable const LinearOperator_SM *op;
        mutable const preconditioner_type *inherited_preconditioner;

        mutable vector_type c, d;
        mutable vector_type g;
        mutable scalar_type alpha;
        mutable scalar_type beta;
        

        Preconditioner_SM(const vector_operations_type *vector_operations_)
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
        }
        void set_vectors(const vector_type& c_, const vector_type& d_, const scalar_type& alpha_) const
        {
            c = c_;
            d = d_;
            alpha = alpha_;
            beta = scalar_type(1);
        }
        void set_vectors(const scalar_type& beta_, const vector_type& c_, const vector_type& d_, const scalar_type& alpha_) const
        {
            c = c_;
            d = d_;
            alpha = alpha_;
            beta = beta_;
        }        
        void set_inherited_preconditioner(const preconditioner_type *inherited_preconditioner_) const
        {
            inherited_preconditioner = inherited_preconditioner_;
        }
        void apply(vector_type& x)const
        {
            //calc: y := mul_x*x
            vector_operations->assign_mul(-1.0/alpha, d, g); // g=-1/alpha*d
            inherited_preconditioner->apply(x); // inv(A)u=>x
            inherited_preconditioner->apply(g); // inv(A)g=>g
            
            scalar_type dot_prod_num = vector_operations->scalar_prod(c, x); // (c,Au):=(c,x)=>cx
            scalar_type dot_prod_din = vector_operations->scalar_prod(c, g); // (c,Ag):=(c,g)=>cg
            scalar_type mul_g=-dot_prod_num/(beta+dot_prod_din);
            // ( x/beta - g/beta*(cx)/(beta+cg) ) -> x

            //calc: y := mul_x*x + mul_y*y
            vector_operations->add_mul(mul_g/beta, g, scalar_type(1/beta), x);

        }

    };

public:
    typedef LinearOperator_SM LinearOperator_SM_t;
    typedef Preconditioner_SM Preconditioner_SM_t;

    typedef LinearSolver<LinearOperator_SM_t, 
                        Preconditioner_SM_t,
                        VectorOperations,
                        Monitor, Log>           linear_solver_type;

    sherman_morrison_linear_system_solve(const Preconditioner *prec_, const VectorOperations *vec_ops_, Log *log_): 
    linear_solver(vec_ops_,log_), 
    prec(vec_ops_), 
    oper(vec_ops_) 
    {
        
        linear_solver.set_preconditioner(&prec);
        prec.set_inherited_preconditioner(prec_);

    }

    ~sherman_morrison_linear_system_solve()
    {

    }


    linear_solver_type *get_linsolver_handle()
    {
        return &linear_solver;
    }



    //solves full extended system with rank 1 update
    bool solve(const LinearOperator &A, const vector_type &c, const vector_type &d, const scalar_type &alpha, const vector_type &b, const scalar_type &beta, vector_type &u, scalar_type &v) const
    {
        bool flag = false;
        
        oper.set_inherited_linear_operator(&A);
        oper.set_vectors(c, d, alpha, b, beta);
        prec.set_vectors(c, d, alpha);
        flag = linear_solver.solve(oper, oper.get_rhs(), u);
        v = oper.get_scalar(u);
        return flag;
    }

    //solves Sherman - Morrison version as: 
    //    (beta A - 1/alpha d c^t)u=b
    bool solve(const scalar_type &beta, const LinearOperator &A, const scalar_type &alpha, const vector_type &c, const vector_type &d,  const vector_type &b, vector_type &u) const
    {
        bool flag = false;
        
        oper.set_inherited_linear_operator(&A);
        oper.set_vectors(beta, c, d, alpha);
        prec.set_vectors(beta, c, d, alpha);
        flag = linear_solver.solve(oper, b, u);
        return flag;
    }

private:
    linear_solver_type linear_solver;
    LinearOperator_SM_t oper;
    Preconditioner_SM_t prec;


};

}
}


#endif