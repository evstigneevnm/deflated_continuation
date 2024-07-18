#ifndef __KURAMOTO_SIVASHINSKIY_2D__
#define __KURAMOTO_SIVASHINSKIY_2D__


/**
*    Problem class for:
*    (1)    lambda*(a_val*(uu_x+uu_y)+L(u))+b_val*B(u)
*
*    solves using Fourier method via FFT
*    Constructor sets up constant scalars "a_val" and "b_val"
*    includes the following methods:
*    - F(x,lambda) solves (1) for given (x,lambda)
*    - set_linearization_point(x0, lambda0) sets point of linearization for calculation of Jacobians
*    - jacobian_u(x) solves Jacobian F_x at (x0, lambda0) for given x using variational formulation
*    - jacobian_lambda(x) solves Jacobian F_lambda at (x0, lambda0) for given x using variational formulation
*    - preconditioner_jacobian_u(dr) applies Jacobi preconditioner for the Jacobian F_x at given (x0, lambda0)
*    axillary:
*    - set_cuda_grid calculates CUDA grid
*    - get_cuda_grid returns calculated grid
*
*
*/
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D_ker.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/plot_solution.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <fstream>

namespace nonlinear_operators
{


template<class FFT_type, class VectorOperations_R, class VectorOperations_C, class VectorOperations_RC_reduced, 
unsigned int BLOCK_SIZE_x=32, unsigned int BLOCK_SIZE_y=16>
class Kuramoto_Sivashinskiy_2D
{
public:
    
    typedef VectorOperations_R vector_operations_real;
    typedef VectorOperations_C vector_operations_complex;
    typedef VectorOperations_RC_reduced vector_operations_real_im;
    typedef typename VectorOperations_R::scalar_type  T;
    typedef typename VectorOperations_R::vector_type  T_vec;
    typedef typename VectorOperations_C::scalar_type  TC;
    typedef typename VectorOperations_C::vector_type  TC_vec;
    typedef typename VectorOperations_RC_reduced::scalar_type  T_im;
    typedef typename VectorOperations_RC_reduced::vector_type  T_vec_im;   

    //class for plotting
    typedef plot_solution<vector_operations_real> plot_t;

    Kuramoto_Sivashinskiy_2D(T a_val_, T b_val_, size_t Nx_, size_t Ny_, 
        vector_operations_real *vec_ops_R_, 
        vector_operations_complex *vec_ops_C_, 
        vector_operations_real *vec_ops_R_im_,
        FFT_type *FFT_): 
    Nx(Nx_), Ny(Ny_), 
    vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), vec_ops_R_im(vec_ops_R_im_),
    FFT(FFT_), 
    a_val(a_val_), b_val(b_val_)
    {
        common_constructor_operation();
        calculate_cuda_grid();
        init_all_derivatives();
    }

    Kuramoto_Sivashinskiy_2D(T a_val_, T b_val_, size_t Nx_, size_t Ny_, dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_, 
        vector_operations_real *vec_ops_R_, 
        vector_operations_complex *vec_ops_C_, 
        vector_operations_real *vec_ops_R_im_,
        FFT_type &FFT_): 
    Nx(Nx_), Ny(Ny_), 
    dimGrid(dimGrid_), dimGrid_F(dimGrid_F_), dimBlock(dimBlock_), 
    vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), vec_ops_R_im(vec_ops_R_im_),
    FFT(FFT_), 
    a_val(a_val_), b_val(b_val_)
    {
        common_constructor_operation();
        init_all_derivatives();
    }

    ~Kuramoto_Sivashinskiy_2D()
    {
        delete plot;

        vec_ops_C->stop_use_vector(gradient_x); vec_ops_C->free_vector(gradient_x);
        vec_ops_C->stop_use_vector(gradient_y); vec_ops_C->free_vector(gradient_y);
        vec_ops_C->stop_use_vector(Laplace); vec_ops_C->free_vector(Laplace);
        vec_ops_C->stop_use_vector(biharmonic); vec_ops_C->free_vector(biharmonic);
        vec_ops_C->stop_use_vector(u_x_hat); vec_ops_C->free_vector(u_x_hat);
        vec_ops_C->stop_use_vector(u_y_hat); vec_ops_C->free_vector(u_y_hat);        
        vec_ops_C->stop_use_vector(b_hat); vec_ops_C->free_vector(b_hat);
        vec_ops_C->stop_use_vector(u_0); vec_ops_C->free_vector(u_0);
        vec_ops_C->stop_use_vector(u_x_hat0); vec_ops_C->free_vector(u_x_hat0);
        vec_ops_C->stop_use_vector(u_y_hat0); vec_ops_C->free_vector(u_y_hat0);
        vec_ops_C->stop_use_vector(mask23); vec_ops_C->free_vector(mask23);


        vec_ops_C->stop_use_vector(u_helper_in); vec_ops_C->free_vector(u_helper_in);
        vec_ops_C->stop_use_vectors(u_helper_out, u_C_helper_fft, u_helper_out_1); vec_ops_C->free_vectors(u_helper_out, u_C_helper_fft, u_helper_out_1);


        vec_ops_R->stop_use_vector(w1_ext); vec_ops_R->free_vector(w1_ext);
        vec_ops_R->stop_use_vector(w2_ext); vec_ops_R->free_vector(w2_ext);
        vec_ops_R->stop_use_vector(w1_ext_0); vec_ops_R->free_vector(w1_ext_0);
        vec_ops_R->stop_use_vector(w2_ext_0); vec_ops_R->free_vector(w2_ext_0);

        vec_ops_R->stop_use_vector(u_ext); vec_ops_R->free_vector(u_ext);
        vec_ops_R->stop_use_vector(u_x_ext); vec_ops_R->free_vector(u_x_ext);
        vec_ops_R->stop_use_vector(u_y_ext); vec_ops_R->free_vector(u_y_ext);
        vec_ops_R->stop_use_vector(u_ext_0); vec_ops_R->free_vector(u_ext_0);
        vec_ops_R->stop_use_vector(u_x_ext_0); vec_ops_R->free_vector(u_x_ext_0);
        vec_ops_R->stop_use_vector(u_y_ext_0); vec_ops_R->free_vector(u_y_ext_0);
        vec_ops_R->stop_use_vector(du_ext); vec_ops_R->free_vector(du_ext);
        vec_ops_R->stop_use_vector(du_x_ext); vec_ops_R->free_vector(du_x_ext);
        vec_ops_R->stop_use_vectors(du_y_ext, u_R_helper_fft); vec_ops_R->free_vectors(du_y_ext, u_R_helper_fft);

    }


    void print_complex_2d(const std::string& file_name, const TC_vec& uC_in)
    {
        std::vector<TC> uC_out_h(Nx*My,0);
        device_2_host_cpy(uC_out_h.data(), uC_in, Nx*My);
        std::ofstream f(file_name, std::ofstream::out);

        for(int j = 0;j<Nx;j++)
        {
            for(int k = 0;k<My;k++)
            {
                f << uC_out_h[I2(j,k,My)] << " ";
            }
            f << std::endl;
        }
        f.close();
    }

    void print_real_2d(const std::string& file_name, const T_vec& uC_in)
    {
        std::vector<T> uC_out_h(Nx*Ny,0);
        device_2_host_cpy(uC_out_h.data(), uC_in, Nx*Ny);
        std::ofstream f(file_name, std::ofstream::out);

        for(int j = 0;j<Nx;j++)
        {
            for(int k = 0;k<Ny;k++)
            {
                f << uC_out_h[I2(j,k,Ny)] << " ";
            }
            f << std::endl;
        }
        f.close();
    }

    void print_vec(const std::string& file_name, const T_vec_im& v_in)
    {
        auto sz = vec_ops_R_im->get_vector_size();
        std::vector<T> out_h(sz, 0);
        device_2_host_cpy(out_h.data(), v_in, sz);
        std::ofstream f(file_name, std::ofstream::out);
        for(int j = 0;j<sz;j++)
        {
            f << out_h[j] << std::endl;
        }
        f.close();

    }

    //nonlinear Kuramoto Sivashinskiy 2D operator:
    //   F(u,alpha)=v
    void F(const TC_vec& u, const T alpha, TC_vec& v)
    {
        
        // print_complex_2d("u_hat.dat", u);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, (const TC_vec&)gradient_x, mask23, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u, (const TC_vec&)gradient_y, mask23, u_y_hat);
        
        vec_ops_C->mul_pointwise(TC(1,0), u, TC(1,0), mask23, u_helper_out);

        ifft(u_x_hat, u_x_ext);
        ifft(u_y_hat, u_y_ext);
        ifft(u_helper_out, u_ext);
        // print_complex_2d("u_x_hat.dat", u_x_hat);
        // print_complex_2d("u_y_hat.dat", u_y_hat);

        // print_real_2d("u_x_ext.dat", u_x_ext);
        // print_real_2d("u_y_ext.dat", u_y_ext);
        // print_real_2d("u_ext.dat", u_ext);


        //z=x*y;
        vec_ops_R->mul_pointwise(1.0, u_x_ext, 1.0, u_ext, w1_ext);
        vec_ops_R->mul_pointwise(1.0, u_y_ext, 1.0, u_ext, w2_ext);

        // print_real_2d("w1_ext.dat", w1_ext);
        // print_real_2d("w2_ext.dat", w2_ext);

        fft(w1_ext, u_x_hat);
        fft(w2_ext, u_y_hat);

        // print_complex_2d("w1_hat.dat", u_x_hat);
        // print_complex_2d("w2_hat.dat", u_y_hat);

        // b_val*biharmonic*u->v
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(b_val), (const TC_vec&)biharmonic, v);
       
        // alpha*Laplace*u->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u, TC(alpha), (const TC_vec&)Laplace, b_hat);
 
        // alpha*a_val*(uu_x)+alpha*a_val*(uu_y)+b_hat->b_hat
        vec_ops_C->add_mul(TC(alpha*a_val), (const TC_vec&) u_x_hat, TC(alpha*a_val), (const TC_vec&) u_y_hat, TC(1.0), b_hat);
       
        // b_hat+v->v
        vec_ops_C->add_mul(TC(1.0), (const TC_vec&)b_hat, v);
    }
    void F(const T_vec_im& u, const T alpha, T_vec_im& v)
    {
        // TC_vec uC_in; //helping vector for R_im to C
        // TC_vec uC_out; //helping vector for R_im to C
        R2C(u, u_helper_in);
        F(u_helper_in, alpha, u_helper_out);
        C2R(u_helper_out, v);
        // vec_ops_R_im->debug_view(v, "f_out.dat");
        // exit(1);
    }

    void fft_test(const T_vec& u_in, TC_vec& u_out)
    {
        fft(u_in, u_out);
        // vec_ops_C->scale(1.0/(Nx*Ny), u_out);
    }
    void ifft_test(const TC_vec& u_in, T_vec& u_out)
    {
        ifft(u_C_helper_fft, u_out);   
    }

    //just a void function to comply with the convergence_strategy.
    void project(T_vec_im& v)
    {
    }
    //just a void function to comply with convergence_strategy.
    T check_solution_quality(const T_vec_im& v)
    {
        return 0;
    }
    //sets (u_0, alpha_0) for jacobian linearization
    //stores alpha_0, u_0, u_ext_0, u_x_ext_0, u_y_ext_0
    //NOTE: u_ext_0, u_x_ext_0 and u_y_ext_0 MUST NOT BE CHANGED!!!
    void set_linearization_point(const TC_vec& u_0_, const T alpha_0_)
    {
        vec_ops_C->assign(u_0_,u_0);
        alpha_0=alpha_0_;
        vec_ops_C->mul_pointwise(TC(1,0), u_0, TC(1,0), mask23, u_helper_in);
        ifft(u_helper_in, u_ext_0);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u_0, (const TC_vec&)gradient_x, mask23, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) u_0, (const TC_vec&)gradient_y, mask23, u_y_hat);
        ifft(u_x_hat,u_x_ext_0);
        ifft(u_y_hat,u_y_ext_0);


    }

    void set_linearization_point(const T_vec_im& u_0_, const T alpha_0_)
    {
        R2C(u_0_, u_helper_in);
        set_linearization_point(u_helper_in, alpha_0_);
    }

    //variational jacobian for 2D KS equations J=dF/du
    //returns vector dv as Jdu->dv, where J(u_0,alpha_0) linearized at (u_0, alpha_0) by set_linearization_point
    void jacobian_u(const TC_vec& du, TC_vec& dv)
    {
        vec_ops_C->mul_pointwise(TC(1,0), du, TC(1,0), mask23, u_helper_out);
        ifft(u_helper_out, du_ext);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) du, (const TC_vec&)gradient_x, mask23, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) du, (const TC_vec&)gradient_y, mask23, u_y_hat);
        ifft(u_x_hat,du_x_ext);
        ifft(u_y_hat,du_y_ext);            
        vec_ops_R->mul_pointwise(1.0, du_x_ext, 1.0, u_ext_0, w1_ext_0);
        vec_ops_R->mul_pointwise(1.0, du_y_ext, 1.0, u_ext_0, w2_ext_0);

        vec_ops_R->mul_pointwise(1.0, u_x_ext_0, 1.0, du_ext, w1_ext);
        vec_ops_R->mul_pointwise(1.0, u_y_ext_0, 1.0, du_ext, w2_ext);

        //du_x*u_0->u_x_hat0
        fft(w1_ext_0,u_x_hat0);
        //du_y*u_0->u_y_hat0
        fft(w2_ext_0,u_y_hat0);
        //u_x*du_0->u_x_hat
        fft(w1_ext,u_x_hat);
        //u_y*du_0->u_y_hat
        fft(w2_ext,u_y_hat);

        // b*d^4du->dv
        // alpha*d^2du->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) du, TC(b_val), (const TC_vec&)biharmonic, dv);
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) du, TC(alpha_0), (const TC_vec&)Laplace, b_hat); 

        //a_val*alpha*(u_x_hat0+u_y_hat0+u_x_hat+u_y_hat)-> u_y_hat
        vec_ops_C->add_mul(TC(1.0), u_x_hat0, TC(1.0), u_y_hat0, 
                            TC(1.0), u_x_hat, TC(1.0), u_y_hat);
        // a_val_alpha*u_y_hat+b_hat+dv->dv
        vec_ops_C->add_mul(TC(a_val*alpha_0), u_y_hat, TC(1.0), b_hat, TC(1.0), dv);

        // print_complex_2d("u_x_hat.dat", u_x_hat);
        // print_complex_2d("u_x_hat.dat", u_y_hat);
        
        // print_complex_2d("df_out.dat", dv);
        // exit(1);        

    }
    void jacobian_u(const T_vec_im& du, T_vec_im& dv)
    {
        R2C(du, u_helper_in);
        jacobian_u(u_helper_in, u_helper_out);
        C2R(u_helper_out, dv);
    }

    //variational jacobian for 2D KS equations J=dF/dalpha
    void jacobian_alpha(TC_vec& dv)
    {
        vec_ops_R->mul_pointwise(1.0, u_x_ext_0, 1.0, u_ext_0, w1_ext_0);
        vec_ops_R->mul_pointwise(1.0, u_y_ext_0, 1.0, u_ext_0, w2_ext_0);
        //du_x0*u_0->u_x_hat0
        fft(w1_ext_0, u_x_hat0);
        //du_y0*u_0->u_y_hat0
        fft(w2_ext_0, u_y_hat0);
        // d^2du->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) u_0, TC(1.0), (const TC_vec&)Laplace, dv);
        //a_val*alpha*(u_x_hat0+u_y_hat0)->u_y_hat0
        vec_ops_C->add_mul(TC(1.0), u_x_hat0, TC(1.0), u_y_hat0);
        // a_val*u_y_hat0+dv->dv
        vec_ops_C->add_mul(TC(a_val), u_y_hat0, TC(1.0,0.0), dv);
    }    
    void jacobian_alpha(const TC_vec& x0, const T& alpha, TC_vec& dv)
    {
            
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) x0, TC(1.0,0.0), (const TC_vec&)gradient_x, u_x_hat);
        vec_ops_C->mul_pointwise(TC(1.0,0.0), (const TC_vec&) x0, TC(1.0,0.0), (const TC_vec&)gradient_y, u_y_hat);
        
        ifft(u_x_hat, u_x_ext);
        ifft(u_y_hat, u_y_ext);
        ifft(dv, u_ext);

        vec_ops_R->mul_pointwise(1.0, u_x_ext, 1.0, u_ext, w1_ext);
        vec_ops_R->mul_pointwise(1.0, u_y_ext, 1.0, u_ext, w2_ext);
        fft(w1_ext, u_x_hat);
        fft(w2_ext, u_y_hat);

        // d^2du->b_hat
        vec_ops_C->mul_pointwise(TC(1.0), (const TC_vec&) x0, TC(1.0), (const TC_vec&)Laplace, b_hat); 

        //a_val*alpha*(u_x_hat0+u_y_hat0)->u_y_hat0
        vec_ops_C->add_mul(TC(1.0,0.0), u_x_hat, TC(1.0,0.0), u_y_hat);
        
        
        vec_ops_C->add_mul(TC(a_val), u_y_hat, TC(1.0), b_hat, TC(1.0), dv);

    }

    void jacobian_alpha(T_vec_im& dv)
    {
        R2C(dv, u_helper_in);
        jacobian_alpha(u_helper_in);
        C2R(u_helper_in, dv);
    }

    void jacobian_alpha(const T_vec_im& x0, const T& alpha, T_vec& dv)
    {
        R2C(x0, u_helper_in);
        jacobian_alpha(u_helper_in, alpha, u_helper_out);
        C2R(u_helper_out, dv);
    }


    void preconditioner_jacobian_u(TC_vec& dr)
    {

        //calc: z := mul_x*x + mul_y*y
        vec_ops_C->assign_mul(TC(b_val),  (const TC_vec&)biharmonic,  TC(alpha_0), (const TC_vec&)Laplace, b_hat); //b_val*biharmonic+lambda*laplace->z
        vec_ops_C->set_value_at_point(TC(1), 0, b_hat);
        //calc: x := x/(mul_y*y)
        vec_ops_C->div_pointwise(dr, TC(1), (const TC_vec&)b_hat); //dr=dr/(1*z);

    }
    void preconditioner_jacobian_u(T_vec_im& dr)
    {
        R2C(dr, u_helper_in);
        preconditioner_jacobian_u(u_helper_in);
        C2R(u_helper_in, dr);
    }

    //preconditioner for the temporal Jacobian a*E + b*dF/du
    void preconditioner_jacobian_temporal_u(TC_vec& dr, T a, T b)
    {
        //calc: z := mul_x*x + mul_y*y
        vec_ops_C->assign_mul(TC(b_val),  (const TC_vec&)biharmonic,  TC(alpha_0), (const TC_vec&)Laplace, b_hat); //b_val*biharmonic+lambda*laplace->z
        vec_ops_C->set_value_at_point(TC(1), 0, b_hat);
        //calc: x := x/(mul_y*y)
        vec_ops_C->div_pointwise(dr, TC(1), (const TC_vec&)b_hat); //dr=dr/(1*z);                
    }
    void preconditioner_jacobian_temporal_u(T_vec_im& dr, T a, T b)
    {
            
        R2C(dr, u_helper_in);
        preconditioner_jacobian_temporal_u(u_helper_in, a, b);
        C2R(u_helper_in, dr);
    }



    void exact_solution(const T alpha, T_vec_im& vec_out)
    {
       vec_ops_R_im->assign_scalar(T(0.0), vec_out);
    }

    void set_cuda_grid(dim3 dimGrid_, dim3 dimGrid_F_, dim3 dimBlock_)
    {
        dimGrid=dimGrid_;
        dimGrid_F=dimGrid_F_;
        dimBlock=dimBlock_;
    }

    void get_cuda_grid(dim3 &dimGrid_, dim3 &dimGrid_F_, dim3 &dimBlock_)
    {  
        dimGrid_=dimGrid;
        dimGrid_F_=dimGrid_F;
        dimBlock_=dimBlock;

    }
    
    void C2R(const TC_vec& vec_C, T_vec_im& vec_R)
    {
        C2R_<TC, T_vec_im, TC_vec>(BLOCK_SIZE_x*BLOCK_SIZE_y, Nx, My, (TC_vec&) vec_C, vec_R);
    }

    void R2C(const T_vec_im& vec_R, TC_vec& vec_C)
    { 
        R2C_<TC, T_vec_im, TC_vec>(BLOCK_SIZE_x*BLOCK_SIZE_y, Nx, My, (T_vec_im&)vec_R, vec_C);
    }
    
    void physical_solution(TC_vec& u_in, T_vec& u_out)
    {
       ifft(u_in, u_out);
    }

    void physical_solution(const T_vec_im& u_in, T_vec& u_out)
    {
       R2C(u_in, u_helper_in);
       physical_solution(u_helper_in, u_out);
    }
    
    void norm_bifurcation_diagram(const T_vec_im& u_in, std::vector<T>& res)
    {
        physical_solution((T_vec_im&)u_in, du_y_ext); //should i use another array??!?? du_y_ext can be bad! Check it!
        T_vec physical_host = vec_ops_R->view(du_y_ext);
        T val1 = physical_host[I2(int(Nx/3.0), int(Ny/3.0), Nx)];
        T val2 = physical_host[I2(int(Nx/5.0), int(2.0*Ny/3.0), Nx)];
        T val3 = physical_host[I2(int(Nx/4.0), int(Ny/5.0), Nx)];
        T val4 = physical_host[I2(int(Nx/2.0), int(Ny/3.0), Nx)];
        T val5 = vec_ops_R_im->norm_l2(u_in);
        
        res.clear();
        res.reserve(5);
        res.push_back(val1);
        res.push_back(val2);
        res.push_back(val3);
        res.push_back(val4);
        res.push_back(val5);

    }

    void fourier_solution(T_vec& u_in, TC_vec& u_out)
    {
       fft(u_in, u_out); 
    }

    void fourier_solution(T_vec& u_in, T_vec_im& u_out)
    {
        fourier_solution(u_in, u_helper_out);
        C2R(u_helper_out, u_out);
    }

    void randomize_vector(T_vec_im& u_out, int steps_ = -1, bool random = true)
    {
        std::srand(unsigned(std::time(0))); //init new seed
        if(random)
        {
            const int NN = 4;            
            vec_ops_R->assign_random(du_x_ext);
            int steps = steps_;
            
            if(steps_ == -1)
            {
                steps = std::rand()%NN + 1;     // random from 1 to NN
            }        


            fourier_solution(du_x_ext, u_helper_out);
            //vec_ops_C->assign_scalar(TC(steps,0), u_helper_out);
            T ww = std::pow(1.0*steps, 1.0/(1.0*steps) );
            T dt = 0.1;
            for(int st=0;st<steps;st++)
            {
                apply_smooth<T, TC>(dimGrid_F, dimBlock, Nx, My, dt,  (TC_vec&)Laplace, u_helper_out, u_helper_out_1);
                vec_ops_C->assign_mul(ww, u_helper_out_1, u_helper_out);
            }
            vec_ops_C->normalize(u_helper_out_1);
            vec_ops_C->scale(steps*Nx*Ny*steps/dt, u_helper_out_1);

            C2R(u_helper_out_1, u_out);
            // T_vec u_out_ph;
            // vec_ops_R->init_vector(u_out_ph); vec_ops_R->start_use_vector(u_out_ph);
            // physical_solution(u_out, u_out_ph);
            // plot->write_to_disk("random_vector.pos", u_out_ph);
            // vec_ops_R->stop_use_vector(u_out_ph); vec_ops_R->free_vector(u_out_ph);
            // exit(1);
        }
        else
        {
            auto exp_func = [](int j, int k, int shift, T muu)
            {
                return exp( -muu*((j-shift)*(j-shift)+(k-shift)*(k-shift)) );
            };
            std::vector<T> uR_in_h(Nx*Ny,0);

            int shift1 = std::max(std::floor(Nx/3), std::floor(Ny/3));
            int shift2 = std::max(std::floor(2*Nx/3), std::floor(2*Ny/3));

            for(int j=0;j<Nx;j++)
            {
                for(int k=0;k<Ny;k++)
                {
                    T muu = 0.05;
                    uR_in_h[I2(j,k,Ny)] = exp_func(j,k,shift1,muu)-exp_func(j,k,shift2,muu);
                }
            }        
            host_2_device_cpy(du_x_ext, uR_in_h.data(), Nx*Ny);
            fft(du_x_ext, u_helper_out);
            print_complex_2d("rand_vec_hat.dat", u_helper_out);
            C2R(u_helper_out, u_out);
            R2C(u_out,u_helper_out);
            print_complex_2d("rand_vec_hat_c2v2c.dat", u_helper_out);
            C2R(u_helper_out, u_out);
            // print_vec("rand_vec.dat", u_out);
            // std::string file_name = "sooth_file_" + std::to_string(steps) + std::string(".pos");
            // write_solution(file_name, u_out);
        }
    }
    
    void write_solution(const std::string& f_name, const T_vec_im& u_in)
    {   
        

        T_vec u_out;
        vec_ops_R->init_vector(u_out); vec_ops_R->start_use_vector(u_out); 
        
        physical_solution(u_in, u_out);

        plot->write_to_disk(f_name, u_out);

        vec_ops_R->stop_use_vector(u_out); vec_ops_R->free_vector(u_out); 

    }
    void write_solution_plain(const std::string& f_name, const T_vec_im& u_in)
    {   
        

        T_vec u_out;
        vec_ops_R->init_vector(u_out); vec_ops_R->start_use_vector(u_out); 
        
        physical_solution(u_in, u_out);

        plot->write_to_disk_plain(f_name, u_out);

        vec_ops_R->stop_use_vector(u_out); vec_ops_R->free_vector(u_out); 

    }




private:
    T a_val,b_val;
    dim3 dimGrid;
    dim3 dimBlock;
    dim3 dimGrid_F;
    vector_operations_real *vec_ops_R;
    vector_operations_complex *vec_ops_C;
    vector_operations_real_im *vec_ops_R_im;

    size_t Nx, Ny, My; //size in physical space Nx*Ny
                       //size in Fourier space Nx*My
                       //size in reduced Fourier space Nx*My-1
    FFT_type *FFT;
    TC_vec gradient_x=nullptr;
    TC_vec gradient_y=nullptr;
    TC_vec Laplace=nullptr;
    TC_vec biharmonic=nullptr;
    TC_vec u_x_hat=nullptr;
    TC_vec u_y_hat=nullptr;
    TC_vec b_hat=nullptr;
    TC_vec u_0=nullptr; // linearization point solution
    TC_vec u_x_hat0=nullptr;
    TC_vec u_y_hat0=nullptr;
    TC_vec u_C_helper_fft = nullptr;
    TC_vec u_helper_out_1 = nullptr;

    TC_vec u_helper_in = nullptr;  //vector for using only R outside
    TC_vec u_helper_out = nullptr;  //vector for using only R outside
    TC_vec mask23 = nullptr;

    T alpha_0=0.0;   // linearization point parameter

    T_vec w1_ext=nullptr;
    T_vec w2_ext=nullptr;
    T_vec w1_ext_0=nullptr;
    T_vec w2_ext_0=nullptr;
    T_vec u_ext=nullptr;
    T_vec u_x_ext=nullptr;
    T_vec u_y_ext=nullptr;

    T_vec u_ext_0=nullptr;
    T_vec u_x_ext_0=nullptr;
    T_vec u_y_ext_0=nullptr;
    T_vec du_ext=nullptr;
    T_vec du_x_ext=nullptr;
    T_vec du_y_ext=nullptr;
    T_vec u_R_helper_fft = nullptr;

    plot_t* plot;

    void common_constructor_operation()
    {  
        My=FFT->get_reduced_size();
        vec_ops_C->init_vector(gradient_x); vec_ops_C->start_use_vector(gradient_x); 
        vec_ops_C->init_vector(gradient_y); vec_ops_C->start_use_vector(gradient_y); 
        vec_ops_C->init_vector(Laplace); vec_ops_C->start_use_vector(Laplace); 
        vec_ops_C->init_vector(biharmonic); vec_ops_C->start_use_vector(biharmonic); 
        vec_ops_C->init_vector(u_x_hat); vec_ops_C->start_use_vector(u_x_hat); 
        vec_ops_C->init_vector(u_y_hat); vec_ops_C->start_use_vector(u_y_hat); 
        vec_ops_C->init_vector(b_hat); vec_ops_C->start_use_vector(b_hat); 
        vec_ops_C->init_vector(u_0); vec_ops_C->start_use_vector(u_0); 
        vec_ops_C->init_vector(u_x_hat0); vec_ops_C->start_use_vector(u_x_hat0); 
        vec_ops_C->init_vector(u_y_hat0); vec_ops_C->start_use_vector(u_y_hat0); 
        vec_ops_C->init_vector(mask23); vec_ops_C->start_use_vector(mask23); 
        vec_ops_C->init_vector(u_helper_in); vec_ops_C->start_use_vector(u_helper_in); 
        vec_ops_C->init_vectors(u_helper_out, u_C_helper_fft, u_helper_out_1); vec_ops_C->start_use_vectors(u_helper_out, u_C_helper_fft, u_helper_out_1);

        vec_ops_R->init_vector(w1_ext); vec_ops_R->start_use_vector(w1_ext); 
        vec_ops_R->init_vector(w2_ext); vec_ops_R->start_use_vector(w2_ext); 
        vec_ops_R->init_vector(w1_ext_0); vec_ops_R->start_use_vector(w1_ext_0); 
        vec_ops_R->init_vector(w2_ext_0); vec_ops_R->start_use_vector(w2_ext_0);         
        vec_ops_R->init_vector(u_ext); vec_ops_R->start_use_vector(u_ext); 
        vec_ops_R->init_vector(u_x_ext); vec_ops_R->start_use_vector(u_x_ext); 
        vec_ops_R->init_vector(u_y_ext); vec_ops_R->start_use_vector(u_y_ext);
        vec_ops_R->init_vector(u_ext_0); vec_ops_R->start_use_vector(u_ext_0); 
        vec_ops_R->init_vector(u_x_ext_0); vec_ops_R->start_use_vector(u_x_ext_0); 
        vec_ops_R->init_vector(u_y_ext_0); vec_ops_R->start_use_vector(u_y_ext_0);
        vec_ops_R->init_vector(du_ext); vec_ops_R->start_use_vector(du_ext);
        vec_ops_R->init_vector(du_x_ext); vec_ops_R->start_use_vector(du_x_ext);
        vec_ops_R->init_vectors(du_y_ext,u_R_helper_fft); vec_ops_R->start_use_vectors(du_y_ext,u_R_helper_fft);
        
        plot = new plot_t(vec_ops_R, Nx, Ny, T(2*3.14159265358979), T(2*3.14159265358979) ) ; 
        form_mask();   
    }

    template<class Vec>
    void build_mask_matrix(int Nx, int Ny, Vec& mask_2_3)
    {

        int m,n;
        for(int k=0;k<Ny;k++)
        {  
            n=k;
            
            for(int j=0;j<Nx;j++){
                m=j; 
                if(j>=Nx/2) m=j-Nx;
                
                T kx=m;
                T ky=n;
                T kxMax=Nx;
                T kyMax=Ny;

                T sphere2 = 4*kx*kx/(kxMax*kxMax)+ky*ky/(kyMax*kyMax); //4 because maxNx/2

                //  2/3 limitation!
                if(sphere2<4.0/9.0)
                {
                    mask_2_3[I2(j,k,Ny)]=TC(1.0,0);
                }
                else
                {
                    mask_2_3[I2(j,k,Ny)]=0.0; //!
                }

            }
        }  
        mask_2_3[0]=0.0;
    }

    void form_mask()
    {
        
        std::vector<TC> mask23_h(Nx*My, 0);
        build_mask_matrix(Nx, My, mask23_h );
        host_2_device_cpy<TC>(mask23, mask23_h.data() , Nx*My);
        // std::vector<T> mask23_rh;
        // for(auto &x: mask23_h)
        // {
        //     mask23_rh.push_back( x.real() );
        // }
        // plot->write_out_file_pos("mask_ks.pos", mask23_rh.data(), Nx, My, 1);
    }


    void init_all_derivatives()
    {
        set_gradient_coefficients();
        set_Laplace_coefficients();
        set_biharmonic_coefficients();        
    }

    void calculate_cuda_grid()
    {
        dim3 s_dimBlock( BLOCK_SIZE_x, BLOCK_SIZE_y );
        dimBlock=s_dimBlock;
        unsigned int blocks_x=floor(Nx/( BLOCK_SIZE_x ))+1;
        unsigned int blocks_y=floor(Ny/( BLOCK_SIZE_y ))+1;
        unsigned int blocks_y_F=floor(My/( BLOCK_SIZE_y ))+1;
        dim3 s_dimGrid( blocks_x, blocks_y);
        dimGrid=s_dimGrid;
        dim3 s_dimGrid_F( blocks_x, blocks_y_F);
        dimGrid_F=s_dimGrid_F;
    }


    void set_gradient_coefficients()
    {
        gradient_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y);
    }

    void set_Laplace_coefficients()
    {
        Laplace_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, Laplace);
    }
   
    void set_biharmonic_coefficients()
    {
        biharmonic_Fourier<TC>(dimGrid_F, dimBlock, Nx, My, gradient_x, gradient_y, biharmonic);
    }

    void ifft(const TC_vec& u_hat_, T_vec& u_)
    {
        vec_ops_C->assign_mul(1, u_hat_, u_C_helper_fft);
        FFT->ifft(u_C_helper_fft, u_);
        T scale = T(1.0)/(Nx*Ny);
        vec_ops_R->scale(scale, u_);
    }
    void fft(const T_vec& u_, TC_vec& u_hat_)
    {
        vec_ops_R->assign_mul(1, u_, u_R_helper_fft);
        FFT->fft(u_R_helper_fft, u_hat_);
        // TC scale = TC((1.0)/(Nx*Ny), 0);
        // vec_ops_C->scale(scale, u_hat_);
    }


    


};

}

#endif
