#ifndef __STABILITY_STABILITY_ANALYSIS_HPP__
#define __STABILITY_STABILITY_ANALYSIS_HPP__

/**
*   The main stability class that basically assembles everything
*   and then executes the particular eigensolver to obtain the dim of the unstable manfold
*   for a provided solution.
*/
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <external_libraries/lapack_wrap.h>
#include <stability/inverse_power_iterations/system_operator_stability.h>
#include <stability/inverse_power_iterations/arnoldi_power_iterations.h>

namespace stability
{


/**
 * @brief      This is a general class for the stability analysis.
 *
 * @tparam     VectorOperations           { N-sized vector operations }
 * @tparam     NonlinearOperations        { General nonlinear operator }
 * @tparam     Log                        { Log class }
 * @tparam     Newton                     { Newton method for the nonlinear operator }
 * @tparam     LinearOperators            { Class that contains all Linear Operators that are called from the eigensolver, including those that are used with transformations  }
 * @tparam     EigenSolver                { description }
 */
template<class VectorOperations, class NonlinearOperations, class Log, class Newton, class EigenSolver>
class stability_analysis
{

public:
    typedef typename VectorOperations::scalar_type T;
    typedef typename VectorOperations::vector_type T_vec;
    using eigs_t = typename EigenSolver::eigs_t;
private:
    using eig_t = typename eigs_t::value_type;

// remove all spesific eigensolver shit out of here.
// we only use eigensolver as an abstract method that returns the set of eigenvalues
// The eigensolver class is passed as the template parameter.
// It should contain the methods:
// void set_linear_operator_stable_eigenvalues_halfplane(int)
// std::vector< std::complex<T> > execute()
// 

public:
    stability_analysis(VectorOperations* vec_ops_, Log* log_, NonlinearOperations* nonlin_op_, Newton* newton_, EigenSolver* eig_solver_):
    vec_ops(vec_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    newton(newton_),
    eig_solver(eig_solver_)
    {
        
        vec_ops->init_vector(x_p1); vec_ops->start_use_vector(x_p1);
        vec_ops->init_vector(x_p2); vec_ops->start_use_vector(x_p2);

    }
    ~stability_analysis()
    {
        vec_ops->stop_use_vector(x_p1); vec_ops->free_vector(x_p1);
        vec_ops->stop_use_vector(x_p2); vec_ops->free_vector(x_p2);        
    }

    void set_linear_operator_stable_eigenvalues_halfplane(const T sign_)
    {
        eig_solver->set_linear_operator_stable_eigenvalues_halfplane(sign_);
    }   
    
    std::pair<int, int> execute(const T_vec& u0_in, const T lambda)
    {
        nonlin_op->set_linerization_point(u0_in, lambda);

        eigs_t eigs = eig_solver->execute();


        //std::ofstream myfile;
        //myfile.open ("eigs.dat");
        //myfile << re << " " << im << std::endl;
        //myfile.close();
        
        int dim_real = count_real(eigs);
        int dim_complex = count_complex(eigs);

        for(auto &x: eigs)
        {
            T re = x.real();
            T im = x.imag();
            if(re>=0.0)
            {
                if(im>=0.0)
                    log->info_f("   %.3lf+%.3lfi", double(re), double(im));
                else
                    log->info_f("   %.3lf%.3lfi", double(re), double(im));
            }

        }
       
        log->info_f("stability.execute: unstable manifold dimentsion at lambda = %lf: real = %i, complex = %i", double(lambda), dim_real, dim_complex);

        std::pair<int, int> unstable_dim = std::make_pair(dim_real, dim_complex);
        return(unstable_dim);
    }



    void bisect_bifurcation_point_known(const T_vec x_1, const T lambda_1, std::pair<int, int> dim1, const T_vec x_2, const T lambda_2, std::pair<int, int> dim2, T_vec& x_p, T& lambda_p, unsigned int max_bisect_iterations = 15)
    {
        int iter = 0;
        T lambda_a = lambda_1;
        T lambda_b = lambda_2;
        vec_ops->assign(x_1, x_p1);
        vec_ops->assign(x_2, x_p2);
        while(iter<max_bisect_iterations)
        {
            lambda_p = lambda_a + T(0.5)*(lambda_b-lambda_a);
            linear_interp_solution(x_p1, x_p2, x_p); //initial guess for x_p

            bool converged = newton->solve(nonlin_op, x_p, lambda_p);
            if(!converged)
            {
                // vec_ops_l->assign(x_p1, x_p);
                // vec_ops_l->add_mul(T(0.333), x_p2, x_p);
                // newton->solve(nonlin_op, x_p, lambda_a);

                // vec_ops_l->assign_scalar(T(0.0), x_p);
                throw std::runtime_error(std::string("stability.bisect_bifurcation_point_known: Newton method failed to converge.") );
            }
            std::pair<int, int> dim_p = execute(x_p, lambda_p);
            if(dim_p == dim2)
            {
                lambda_b = lambda_p;
                vec_ops->assign(x_p, x_p2);
            }
            else if(dim_p == dim1)
            {
                lambda_a = lambda_p;
                vec_ops->assign(x_p, x_p1);                    
            }
            else
            {
                //assume that a new point with dim_p!=dim2 and dim1 is a new dim2 point. This way we can isolate a prticular point if there are more than one bifurcaiton points in the parameter segement.
                dim2 = dim_p;
                lambda_b = lambda_p;
                vec_ops->assign(x_p, x_p2);                    
            }
            iter++;
            log->info_f("stability.bisect_bifurcaiton_point: bisected to lambda = %lf, dim(U) = (%i,%i), iteration = %i", lambda_p, dim_p.first, dim_p.second, iter);  
        }        

    }

    void bisect_bifurcaiton_point(const T_vec& x_1, const T& lambda_1, const T_vec& x_2, const T& lambda_2, T_vec& x_p, T& lambda_p, unsigned int max_bisect_iterations = 15)
    {
        std::pair<int, int> dim1 = execute(x_1, lambda_1);
        std::pair<int, int> dim2 = execute(x_2, lambda_2);
        if(dim1 == dim2)
        {
            log->info_f("stability.bisect_bifurcaiton_point: at %lf dum(U) = (%i,%i), at %lf dim(U) = (%i,%i)", lambda_1, dim1.first, dim1.second, lambda_2, dim2.first, dim2.second);  
            log->info("Nothing to do. Output is not set.");
        }
        else
        {
            bisect_bifurcation_point_known(x_1, lambda_1, dim1, x_2, lambda_2, dim2, x_p, lambda_p, max_bisect_iterations);
        }

    }


private:
    //passed:
    Log* log;
    VectorOperations* vec_ops;
    NonlinearOperations* nonlin_op;
    Newton* newton;
    EigenSolver* eig_solver;
//  created_locally:


    size_t small_rows;
    size_t small_cols;

    T_vec x_p1;
    T_vec x_p2;

    template<class T>
    void delete_if_not_null(T* ptr)
    {
        if(ptr!=nullptr)
            delete ptr;
    }

    int count_real(const eigs_t& elems) 
    {
        int reals_only = std::count_if(elems.begin(), elems.end(), [](eig_t c){return (c.real() > 0)&&(std::abs(c.imag())<1.0e-7);});
        return(reals_only);
    }   
    int count_complex(const eigs_t& elems) 
    {
        int complex_only = std::count_if(elems.begin(), elems.end(), [](eig_t c){return (std::abs(c.imag() )>=1.0e-6)&&(c.real() > 0); });
        return(complex_only/2);
    } 

    void linear_interp_solution(const T_vec& x1, const T_vec& x2, T_vec& x_r)
    {
        //calc: z := mul_x*x + mul_y*y
        vec_ops->assign_mul(T(0.5), x1, T(0.5), x2, x_r);

    }


};

}

#endif // __STABILITY_STABILITY_HPP__