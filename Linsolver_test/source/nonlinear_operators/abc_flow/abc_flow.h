#ifndef __ABC_FLOW_H__
#define __ABC_FLOW_H__


#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <common/macros.h>
#include <nonlinear_operators/abc_flow/abc_flow_ker.h>
#include <nonlinear_operators/abc_flow/plot_solution.h>

#include <common/vector_wrap.h>
#include <common/vector_pool.h>

#include <tuple>
#include <limits>
#include <utility>


namespace nonlinear_operators
{

// 0: fft->scale->ifft
// 1: fft->ifft->scale
// 2: fft->sqrt(scale)->ifft->sqrt(scale)
const char abc_flow_fft_rescale = 0;

template<class VecOps> //it is assumed either over C or over R
class block_vec //axilary class for block vectors
{
public:
    typedef VecOps vector_operations;    
    typedef typename VecOps::vector_type vector_type;
    typedef typename VecOps::scalar_type scalar_type;
private:
    typedef vector_type T_vec;
    typedef scalar_type T;

    VecOps* vec_ops;
    bool allocated = false;
    void set_op(VecOps* vec_ops_){ vec_ops = vec_ops_; }

public:

    block_vec()
    {}
    ~block_vec()
    {
        free();
    }

    void alloc(VecOps* vec_ops_)
    {
        set_op(vec_ops_);
        if(!allocated)
        {
            vec_ops->init_vector(x); vec_ops->start_use_vector(x); 
            vec_ops->init_vector(y); vec_ops->start_use_vector(y); 
            vec_ops->init_vector(z); vec_ops->start_use_vector(z);
            allocated = true;         
        }
    }
    void free()
    {
        if(allocated)
        {
            vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
            vec_ops->stop_use_vector(y); vec_ops->free_vector(y);
            vec_ops->stop_use_vector(z); vec_ops->free_vector(z);
            allocated = false;
        }
    }

    T_vec x = nullptr;
    T_vec y = nullptr;
    T_vec z = nullptr;
};


// VecOpsR sized Nx*Ny*Nz;VecOpsC sized Nx*Ny*Mz; VecOps is the main vector
// operations sized 3*(Nx*Ny*Mz-1)
//
// @tparam     FFT_type      { description }
// @tparam     VecOpsR       { description }
// @tparam     VecOpsC       { description }
// @tparam     VecOps        { description }
// @tparam     BLOCK_SIZE_x  { description }
// @tparam     BLOCK_SIZE_y  { description }
/*
*    Problem class for the ABC flow:
*    (1)    $(U, \nabla U) + \nabla p - \nu \nabla^2 U - f = 0$,
*    (2)    $\nabla \cdot U = 0$,
*    where U=(ux, uy, uz)^T is a 3D vector-function, p is a scalar-function and f is a given vector-funciton RHS: 
*    (3)    $f = ( sin(z) + cos(y); sin(x)+cos(z); sin(y)+cos(x) )^T$.
*    Computaitonal domain is: 
*    (4)    $\Omega := [0;2\pi]\times[0;2\pi]\time[0;2\pi]$.
*    Parameter \lambda: Reynolds number $R = (\nu)^{-1}$.
*    
*    WARNING: pressure is used as a gauge so no explicit pressure is avaliable in the system.
*             therefore the system being solved in reality is:
*    (5)      $P[ (U, \nabla U) - \nu \nabla^2 U - f ] = 0$,
*             where P is the projection operator to div. free vector field:
*    (6)      $P:= (id - \nabla (\nabla)^{-1}) \nabla \cdot)$
*
*    solves using Fourier method via FFT
*    includes the following methods:
*    - project(u, v) projects a vector 'U' and returns vector 'V', s.t. \nabla \cdot V = 0, where u is an 
*  augmented vector, U is a full block vector, i.e. u->U: U={ux,uy,uz}.
* 
*    - F(x,lambda) solves (1) for given (x,lambda)
*    - set_linearization_point(x0, lambda0) sets point of linearization for calculation of Jacobians
*    - jacobian_u(x) solves Jacobian F_x at (x0, lambda0) for given x using variational formulation
*    - jacobian_lambda(x) solves Jacobian F_lambda at (x0, lambda0) for given x using variational formulation
*    - 
*    - preconditioner_jacobian_u(dr) applies Jacobi preconditioner for the Jacobian F_x at given (x0, lambda0)
*    - preconditioner_jacobian_temporal_u(T_vec& dr, T a, T b) applies a+b*J preconditioner, where J is the Jacobian F_x at given (x0, lambda0) and the real values 'a' and 'b' are set.
*    
*    axillary:
*    - set_cuda_grid calculates CUDA grid
*    - get_cuda_grid returns calculated grid
*/
template<class FFT_type, class VecOpsR, class VecOpsC, class VecOps,  
unsigned int BLOCK_SIZE_x = 32, unsigned int BLOCK_SIZE_y = 16>
class abc_flow
{
private:
    typedef VecOpsR vec_R_t;
    typedef VecOpsC vec_C_t;
    typedef VecOps vec_ops_t;
    typedef typename VecOpsR::scalar_type  TR;
    typedef typename VecOpsR::vector_type  TR_vec;
    typedef typename VecOpsC::scalar_type  TC;
    typedef typename VecOpsC::vector_type  TC_vec;
    typedef typename VecOps::scalar_type  T;
    typedef typename VecOps::vector_type  T_vec;   
    //class for plotting
    typedef plot_solution<vec_R_t> plot_t;

    typedef block_vec<vec_R_t> BR_vec;
    typedef block_vec<vec_C_t> BC_vec;
    //vector wraps:    
    typedef vector_wrap<vec_R_t> R_vec;
    typedef vector_wrap<vec_C_t> C_vec;

    //vector pools:
    typedef vector_pool<R_vec> pool_R_t;
    typedef vector_pool<C_vec> pool_C_t;
    typedef vector_pool<BR_vec> pool_BR_t;
    typedef vector_pool<BC_vec> pool_BC_t;


    //calss for low level CUDA kernels
    typedef abc_flow_ker<TR, TR_vec, TC, TC_vec> kern_t;
    




private:
    unsigned int BLOCK_SIZE = BLOCK_SIZE_x*BLOCK_SIZE_y;

    //data all passed to constructor as pointers
    vec_R_t *vec_ops_R;
    vec_C_t *vec_ops_C;
    vec_ops_t *vec_ops;
    size_t Nx, Ny, Nz, Mz; //size in physical space Nx*Ny*Nz
                           //size in Fourier space Nx*Ny*Mz
                           //size in the main stretched vector 3*(Nx*Ny*Mz-1)
    FFT_type *FFT;
    kern_t* kern;
    plot_t* plot;

    //parameters of the external forcing and its magnitude
    //should be passed as parameters???
    T scale_force;

    T Lx;
    T Ly;
    T Lz;    

public:
    abc_flow(size_t Nx_, size_t Ny_, size_t Nz_,
        vec_R_t* vec_ops_R_, 
        vec_C_t* vec_ops_C_, 
        vec_R_t* vec_ops_,
        FFT_type* FFT_): 
    Nx(Nx_), Ny(Ny_), Nz(Nz_), 
    vec_ops_R(vec_ops_R_), vec_ops_C(vec_ops_C_), vec_ops(vec_ops_),
    FFT(FFT_),
    A_(1.0),
    B_(1.0),
    C_(1.0)
    {
        scale_force = T(1.0);

        Mz=FFT->get_reduced_size();
        common_constructor_operation();
        kern = new kern_t(Nx, Ny, Nz, Mz, A_, B_, C_, BLOCK_SIZE_x, BLOCK_SIZE_y);
        init_all_derivatives();
        Lx = T(2.0)*M_PI;
        Ly = T(2.0)*M_PI;
        Lz = T(2.0)*M_PI;
        plot = new plot_t(vec_ops_R_, Nx_, Ny_, Nz_, Lx, Ly, Lz);

        std::cout << "abc_flow===> nonlinear problem class self-testing:" << std::endl;
        for(unsigned int j=0;j<1;j++)
        {   
            auto manufactured_norm = manufactured_solution(j);
            std::cout << "abc_flow===> manufactured solution number: " << j <<  ", solution norm = "<< manufactured_norm.first << ", residual norm = " << manufactured_norm.second << std::endl;
            if( !std::isfinite(manufactured_norm.second) ||(manufactured_norm.second > std::numeric_limits<T>::epsilon()*1000.0 ))
            {
                throw std::runtime_error("manufactures solution number: " + std::to_string(j) + " returned bad residual norm!" );
            }
        }
    }


    ~abc_flow()
    {
        common_distructor_operations();
        delete kern;
        delete plot;
    }

    void F(const T time_p, const T_vec& u, const T Reynolds_, T_vec& v)
    {
        //for general call to the time stepper.
        F(u, Reynolds_, v);
    }

    //   F(u,alpha)=v
    void F(const T_vec& u, const T Reynolds_, T_vec& v)
    {
        BC_vec* U = pool_BC.take(); //input
        BC_vec* W = pool_BC.take(); //output        

        V2C(u, *U);
        F(*U, Reynolds_, *W); 
        C2V(*W, v);

        pool_BC.release(U);
        pool_BC.release(W);
    }
    //   F(u,alpha)=v
    void F(BC_vec& U, const T Reynolds_, BC_vec& W)
    {
        project(U);
        // vec_ops_C->assign_scalar(0.0,W.x);
        // vec_ops_C->assign_scalar(0.0,W.y);
        // vec_ops_C->assign_scalar(0.0,W.z);
        B_V_nabla_F(U, U, W); //W:= (U, nabla) U
        kern->add_mul3(TC(-1.0/Reynolds_,0), forceABC.x, forceABC.y, forceABC.z, W.x, W.y, W.z); // force:= W-force
        project(W); // W:=P[W]??
        kern->negate3(W.x, W.y, W.z); //W:=-W;
        // U := \nabla^2 U
        kern->apply_Laplace(TR(1.0/Reynolds_), Laplace, U.x, U.y, U.z);
        // W := W + U
        kern->add_mul3(TC(1.0,0), U.x, U.y, U.z, W.x, W.y, W.z);   
        
    }

    //sets (u_0, alpha_0) for jacobian linearization
    //stores alpha_0, u_0  MUST NOT BE CHANGED!!!
    void set_linearization_point(const T_vec& u_0_, const T Reynolds_0_)
    {
        
        V2C(u_0_, U_0);   
        project(U_0); //just in case!
        Reynolds_0 = Reynolds_0_;     
    }


    //variational jacobian J=dF/du:=
    //returns vector dv as Jdu->dv, where J(u_0,alpha_0) linearized at (u_0, alpha_0) by set_linearization_point
     void jacobian_u(BC_vec& dU, BC_vec& dV)
    {
    // dV:=P[-(U_0, \nabla) dU - (dU, \nabla) U0] + nu*\nabla^2 dU
        BC_vec* dW = pool_BC.take(); //input        

        // vec_ops_C->assign_scalar(0.0,dW->x);
        // vec_ops_C->assign_scalar(0.0,dW->y);
        // vec_ops_C->assign_scalar(0.0,dW->z);
        project(dU);
        B_V_nabla_F(U_0, dU, dV); //dV:= (dU, nabla) U_0
        B_V_nabla_F(dU, U_0,*dW); //dW:= (U_0, nabla) dU
        kern->add_mul3(TC(1.0,0), dW->x, dW->y, dW->z, dV.x, dV.y, dV.z); // dV:=dV+dW

        kern->negate3(dV.x, dV.y, dV.z); // dV:=-dV
        project(dV); // dV:=P[dV]
        // vec_ops_C->assign_scalar(0.0,dV.x);
        // vec_ops_C->assign_scalar(0.0,dV.y);
        // vec_ops_C->assign_scalar(0.0,dV.z);
        kern->apply_Laplace(TR(1.0/Reynolds_0), Laplace, dU.x, dU.y, dU.z); // dU:=nu*\nabla^2 dU
        kern->add_mul3(TC(1.0,0), dU.x, dU.y, dU.z, dV.x, dV.y, dV.z); // dV:=dU+dV
        pool_BC.release(dW); 
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {

        BC_vec* dU = pool_BC.take(); //input
        BC_vec* dV = pool_BC.take(); //output           
        
        V2C(du, *dU);
        jacobian_u(*dU, *dV);
        C2V(*dV, dv);
        
        pool_BC.release(dU);
        pool_BC.release(dV);

    }

    //variational jacobian J=dF/dalpha
    void jacobian_alpha(BC_vec& dV)
    {
        kern->copy3(U_0.x, U_0.y, U_0.z, dV.x, dV.y, dV.z);
        kern->apply_Laplace(TR(-1.0/(Reynolds_0*Reynolds_0)), Laplace, dV.x, dV.y, dV.z);
        kern->add_mul3(TC(-1.0/(Reynolds_0*Reynolds_0),0), forceABC.x, forceABC.y, forceABC.z, dV.x, dV.y, dV.z);
    } 
  
    void jacobian_alpha(const BC_vec& U0, const T& Reynolds_0_, BC_vec& dV)
    {
          
        kern->copy3(U0.x, U0.y, U0.z, dV.x, dV.y, dV.z);
        kern->apply_Laplace(TR(-1.0/(Reynolds_0_*Reynolds_0_)), Laplace, dV.x, dV.y, dV.z);
        kern->add_mul3(TC(-1.0/(Reynolds_0_*Reynolds_0_),0), forceABC.x, forceABC.y, forceABC.z, dV.x, dV.y, dV.z);

    }

    void jacobian_alpha(T_vec& dv)
    {
        BC_vec* dV = pool_BC.take();
        V2C(dv, *dV);
        jacobian_alpha(*dV); 
        C2V(*dV, dv);
        pool_BC.release(dV);
        project(dv);
    }

    void jacobian_alpha(const T_vec& u0, const T& Reynolds_0_, T_vec& dv)
    {
        BC_vec* dV = pool_BC.take();
        BC_vec* U0 = pool_BC.take();
        V2C(dv, *dV);
        V2C(u0, *U0);
        jacobian_alpha(*U0, Reynolds_0_, *dV);
        pool_BC.release(U0);
        C2V(*dV, dv);
        pool_BC.release(dV);

    }

    //preconditioner for the Jacobian dF/du
    void preconditioner_jacobian_u(BC_vec& dR)
    {
        project(dR);
        kern->apply_iLaplace3(Laplace, dR.x, dR.y, dR.z, Reynolds_0);


        //project(dR);
    }
    void preconditioner_jacobian_u(T_vec& dr)
    {
        BC_vec* dR = pool_BC.take();
        
        V2C(dr, *dR);
        preconditioner_jacobian_u(*dR);
        C2V(*dR, dr);
        
        pool_BC.release(dR);
    }



    //preconditioner for the temporal Jacobian a*E + b*dF/du
    void preconditioner_jacobian_temporal_u(BC_vec& dR, T a, T b)
    {
        project(dR);
        kern->apply_iLaplace3_plus_E(Laplace, dR.x, dR.y, dR.z, Reynolds_0, a, b);
        project(dR);

    }
    void preconditioner_jacobian_temporal_u(T_vec& dr, T a, T b)
    {
        BC_vec* dR = pool_BC.take();
        
        V2C(dr, *dR);
        preconditioner_jacobian_temporal_u(*dR, a, b);
        C2V(*dR, dr);
        
        pool_BC.release(dR);
    }

    //funciton that returns std::vector with different bifurcatoin norms

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res, bool clear_vector = false)
    {
        R_vec* uR0 = pool_R.take();

        physical_solution(u_in, uR0->x); 
        //this is to be changed to kernel in kern class!
        //don't copy the whole vector!
        T_vec physical_host = vec_ops_R->view(uR0->x);
        T val1 = physical_host[I3(int(Nx/3.0), int(Ny/3.0), int(Nz/3.0))];
        T val2 = physical_host[I3(int(Nx/5.0), int(2.0*Ny/3.0), int(1.0*Nz/3.0))];
        T val3 = physical_host[I3(int(Nx/4.0), int(Ny/5.0), int(Nz/5.0))];
        T val4 = physical_host[I3(int(Nx/2.0)-1, int(Ny/2.0)-1, int(Nz/2)-1)];
        T val5 = vec_ops->norm_l2(u_in);
        T val6 = norm(u_in);
        T val7 = vec_ops_R->norm_l2(uR0->x);
        // T val7_x = vec_ops_R->norm_l2(uR0->x);
        // T val7_y = vec_ops_R->norm_l2(uR0->y);
        // T val7_z = vec_ops_R->norm_l2(uR0->z);
        // T val7 = std::sqrt(val7_x*val7_x+val7_y*val7_y+val7_z*val7_z);
        if(clear_vector)
        {
            res.clear();
            res.reserve(7);
        }
        res.push_back(val1); //2
        res.push_back(val2); //3
        res.push_back(val3); //4
        res.push_back(val4); //5
        res.push_back(val5); //6
        res.push_back(val6); //7
        res.push_back(val7); //8

        pool_R.release(uR0);
    }

    //function that returns exact solution(ES)
    //if the ES is trivial, then return zero vector
    void exact_solution_sin_cos(const T& Reynolds, T_vec& u_out)
    {
        BC_vec* UA = pool_BC.take();
        vec_ops_C->assign_mul(TC(scale_force,0), force.x, UA->x);
        vec_ops_C->assign_mul(TC(scale_force,0), force.y, UA->y);
        vec_ops_C->assign_mul(TC(scale_force,0), force.z, UA->z);        
        C2V(*UA, u_out);
        pool_BC.release(UA);        
    }

    void exact_solution_abc(const T& Reynolds, T_vec& u_out)
    {
        BC_vec* UA = pool_BC.take();
        vec_ops_C->assign_mul(TC(scale_force,0), forceABC.x, UA->x);
        vec_ops_C->assign_mul(TC(scale_force,0), forceABC.y, UA->y);
        vec_ops_C->assign_mul(TC(scale_force,0), forceABC.z, UA->z);        
        C2V(*UA, u_out);
        pool_BC.release(UA);        
    }

    void exact_solution(const T& Reynolds, T_vec& u_out)
    {       
        exact_solution_abc(Reynolds, u_out);
    }


    void physical_solution(const T_vec& u_in, TR_vec& u_out)
    {
        
        BC_vec* UC0 = pool_BC.take();
        BR_vec* UR0 = pool_BR.take();

        V2C(u_in, *UC0);
        ifft(*UC0, *UR0);
        kern->apply_abs(UR0->x, UR0->y, UR0->z, u_out);

        pool_BC.release(UC0);
        pool_BR.release(UR0);
    }

    void physical_solution(const T_vec& u_in, TR_vec& u_out_x, TR_vec& u_out_y, TR_vec& u_out_z)
    {
        BC_vec* UC0 = pool_BC.take();
        BR_vec* UR0 = pool_BR.take();

        V2C(u_in, *UC0);
        ifft(*UC0, *UR0);
        vec_ops_R->assign(UR0->x, u_out_x);
        vec_ops_R->assign(UR0->y, u_out_y);
        vec_ops_R->assign(UR0->z, u_out_z);

        pool_BC.release(UC0);
        pool_BR.release(UR0);

    }

    //this is basically a hack, and a nasty one.
    void write_solution_scaled(FFT_type* FFT_vis, size_t Nx_dest, size_t Ny_dest, size_t Nz_dest, const std::string& f_name_vec, const std::string& f_name_abs, const T_vec& u_in)
    {   
        BC_vec* UC0 = pool_BC.take();
        
        size_t Mz_dest = FFT_vis->get_reduced_size();

        V2C(u_in, *UC0);
        TC_vec UX_hat = device_allocate<TC>(Nx_dest, Ny_dest, Mz_dest);
        TC_vec UY_hat = device_allocate<TC>(Nx_dest, Ny_dest, Mz_dest);
        TC_vec UZ_hat = device_allocate<TC>(Nx_dest, Ny_dest, Mz_dest);

        TR_vec UX = device_allocate<TR>(Nx_dest, Ny_dest, Nz_dest);
        TR_vec UY = device_allocate<TR>(Nx_dest, Ny_dest, Nz_dest);
        TR_vec UZ = device_allocate<TR>(Nx_dest, Ny_dest, Nz_dest);

        TR scale=TR(1.0)*Nx_dest*Ny_dest*Nz_dest/(TR(1.0)*Nx*Ny*Nz);

        kern->convert_size(Nx_dest, Ny_dest, Mz_dest, scale, UC0->x, UC0->y, UC0->z, UX_hat, UY_hat, UZ_hat);
        pool_BC.release(UC0);

        T scale_loc = T(1.0)/(Nx_dest*Ny_dest*Nz_dest);

        FFT_vis->ifft(UX_hat, UX);
        FFT_vis->ifft(UY_hat, UY);
        FFT_vis->ifft(UZ_hat, UZ);
        kern->apply_scale_inplace(Nx_dest, Ny_dest, Nz_dest, 1.0/scale, UX, UY, UZ);
        
        TR_vec U_abs = device_allocate<TR>(Nx_dest, Ny_dest, Nz_dest);

        kern->apply_abs(Nx_dest, Ny_dest, Nz_dest, UX, UY, UZ, U_abs);

        plot_t plot_dest(vec_ops_R, Nx_dest, Ny_dest, Nz_dest, Lx, Ly, Lz);
        plot_dest.write_to_disk(f_name_abs, U_abs, 2 );
        plot_dest.write_to_disk(f_name_vec, UX, UY, UZ, 2);

        device_deallocate(UX_hat);
        device_deallocate(UY_hat);
        device_deallocate(UZ_hat);
        device_deallocate(UX);
        device_deallocate(UY);
        device_deallocate(UZ);
        device_deallocate(U_abs);



    }

    void write_solution_abs(const std::string& f_name, const T_vec& u_in)
    {   
        BC_vec* UC0 = pool_BC.take();
        BR_vec* UR0 = pool_BR.take();
        R_vec* u_out = pool_R.take();
        
        V2C(u_in, *UC0);
        ifft(*UC0, *UR0);        
        kern->apply_abs(UR0->x, UR0->y, UR0->z, u_out->x);
        pool_BC.release(UC0);
        pool_BR.release(UR0);

        plot->write_to_disk(f_name, u_out->x, 2 );

        pool_R.release(u_out);
    }

    void write_solution_vec(const std::string& f_name, const T_vec& u_in)
    {   
        
        BR_vec* UR0 = pool_BR.take();

        physical_solution(u_in, UR0->x, UR0->y, UR0->z);

        plot->write_to_disk(f_name, UR0->x, UR0->y, UR0->z, 2);

        pool_BR.release(UR0);

    }


    void B_ABC_exact_solution(T_vec u_out)
    {
        BR_vec* UR = pool_BR.take();
        kern->B_ABC_exact(1.0, UR->x, UR->y, UR->z);

        BC_vec* UC = pool_BC.take();
        fft(*UR, *UC);
        project(*UC);
        C2V(*UC, u_out);
        pool_BR.release(UR);
        pool_BC.release(UC);
    }

    void B_ABC_approx_solution(T_vec u_out)
    {
        BC_vec* UC = pool_BC.take();
        B_V_nabla_F(forceABC, forceABC, *UC);
        project(*UC);
        C2V(*UC, u_out);
        pool_BC.release(UC);
    } 


    void randomize_vector(T_vec u_out, int steps_ = -1)
    {
        
        BC_vec* UC0 = pool_BC.take();
        BR_vec* UR0 = pool_BR.take();

        vec_ops_R->assign_random( UR0->x );
        vec_ops_R->assign_random( UR0->y );
        vec_ops_R->assign_random( UR0->z );
        
        // vec_ops_R->scale(10.0, UR0->x);
        // vec_ops_R->scale(10.0, UR0->y);
        // vec_ops_R->scale(10.0, UR0->z);

        int steps = steps_;
        if(steps_ == -1)
        {
            std::srand(unsigned(std::time(0))); //init new seed
            steps = std::rand()%10 + 1;     // random from 1 to 10

        }

        fft(*UR0, *UC0);
        // imag_vec(*UC0);
        project(*UC0);
        for(int st=0;st<steps;st++)
        {
            smooth(TR(0.05), *UC0);
        }
        project(*UC0);

        C2V(*UC0, u_out);

        pool_BC.release(UC0);
        pool_BR.release(UR0);     

        // std::cout << "div norm = " << div_norm(u_out);
        // write_solution_vec("vec_i.pos", u_out);
    }
    
    
    //tests the method with the some pre-defined manufactured solutions
    //a particular solution is selected by the prived number
    //returns a residual norm
    std::pair<T,T> manufactured_solution(unsigned int solution_number = 0)
    {
        T resid_norm = -1.0;
        T solution_norm = 0.0;
        if(solution_number == 0)
        {
            T Reynolds = 5.0;
            T_vec u_manufactured, resid;
            vec_ops->init_vectors(u_manufactured, resid); vec_ops->start_use_vectors(u_manufactured, resid); 
            exact_solution(Reynolds, u_manufactured);
            solution_norm = vec_ops->norm_l2(u_manufactured);
            F(u_manufactured, Reynolds, resid);
            resid_norm = vec_ops->norm_l2(resid)/solution_norm;
            vec_ops->stop_use_vectors(u_manufactured, resid); vec_ops->free_vectors(u_manufactured, resid); 
        }
        return {solution_norm, resid_norm};
    }


    T check_solution_quality(const T_vec& u_in)
    {
        return div_norm(u_in)/std::max(norm(u_in), 1.0);
    }

    T div_norm(const T_vec& u_in)
    {

        C_vec* uC0 = pool_C.take();
        BC_vec* UC0 = pool_BC.take();

        V2C(u_in, *UC0);
        div(*UC0, uC0->x);
        
        TR norm = vec_ops_C->norm_l2(uC0->x);

        pool_C.release(uC0);
        pool_BC.release(UC0);

        return(norm);

    }

    //projects vector to div free space
    void project(T_vec& u_)
    {
        BC_vec* UC0 = pool_BC.take();
        V2C(u_, *UC0);        
        project(*UC0);
        C2V(*UC0, u_);
        pool_BC.release(UC0);          
    }

    //finds sutable harmonics and uses them to translate the solution is such way 
    //that imaginary part of the selected harminics is zero.
    void translation_fix(const T_vec& u_in_, T_vec& u_out_)
    {
        BC_vec* UC0 = pool_BC.take();
        BC_vec* UC1 = pool_BC.take();
        V2C(u_in_, *UC0);  
        auto res = shift_phases(*UC0);
        auto varphi_x = std::get<0>(res);
        auto varphi_y = std::get<1>(res);
        auto varphi_z = std::get<2>(res);
        translate_solution(*UC0, varphi_x, varphi_y, varphi_z, *UC1);
        C2V(*UC1, u_out_);
        pool_BC.release(UC1);
        pool_BC.release(UC0);

    }


    //translates the whole vector solution in direction, given by varphi_x, varphi_y, varphi_z
    //returns x_translate_ vector that is transalted
    void translate_solution(const T_vec& x_, TR varphi_x, TR varphi_y, TR varphi_z, T_vec& x_translate_)
    {
        BC_vec* UC0 = pool_BC.take();
        BC_vec* UC1 = pool_BC.take();
        V2C(x_, *UC0);  
        translate_solution(*UC0, varphi_x, varphi_y, varphi_z, *UC1);
        C2V(*UC1, x_translate_);
        pool_BC.release(UC1);
        pool_BC.release(UC0);
    }

    //enforces hermitian_symmetry (for real-valued functions in Fourier domain)
    void hermitian_symmetry(const T_vec& x_, T_vec& x_symm_)
    {
        BC_vec* UC0 = pool_BC.take();
        BC_vec* UC1 = pool_BC.take();
        V2C(x_, *UC0);  
        hermitian_symmetry(*UC0, *UC1);
        C2V(*UC1, x_symm_);
        pool_BC.release(UC1); 
        pool_BC.release(UC0);        

    }


    TR norm(const T_vec& u_in_)
    {
        BC_vec* UC1 = pool_BC.take();
        V2C(u_in_, *UC1);  
        auto norm_ = norm(*UC1);
        pool_BC.release(UC1);
        return norm_;
    }
    
private:
    
    //WARNING!
    //These arrays are under low level operations,
    //to reduce memory consumption.
    TC_vec grad_x = nullptr;
    TC_vec grad_y = nullptr;
    TC_vec grad_z = nullptr;
    //WARNING ENDS
    TC_vec Laplace = nullptr;

    //linearized solution point:
    BC_vec U_0;
    T Reynolds_0;
    //vector pools:
    
    pool_R_t pool_R;
    pool_C_t pool_C;    
    pool_BR_t pool_BR;
    pool_BC_t pool_BC;


    BC_vec force;
    BC_vec forceABC;
    BR_vec forceABC_R;

    TC_vec mask23;
    TR A_, B_, C_; //ABC flow constants

    void common_distructor_operations()
    {
        device_deallocate<TC>(grad_x);
        device_deallocate<TC>(grad_y);
        device_deallocate<TC>(grad_z);

        vec_ops_C->stop_use_vector(Laplace);  vec_ops_C->free_vector(Laplace); 
        vec_ops_C->stop_use_vector(mask23);  vec_ops_C->free_vector(mask23);         
        force.free();
        forceABC.free();
        forceABC_R.free();
        U_0.free();

        pool_R.free_all();
        pool_C.free_all();    
        pool_BR.free_all();
        pool_BC.free_all();



    }

    void common_constructor_operation()
    {  
        
        grad_x = device_allocate<TC>(Nx);
        grad_y = device_allocate<TC>(Ny);
        grad_z = device_allocate<TC>(Mz);

        vec_ops_C->init_vector(Laplace);  vec_ops_C->start_use_vector(Laplace); 
        vec_ops_C->init_vector(mask23);  vec_ops_C->start_use_vector(mask23);
        force.alloc(vec_ops_C);
        forceABC.alloc(vec_ops_C);
        forceABC_R.alloc(vec_ops_R);
        U_0.alloc(vec_ops_C);

        //adjust pool sizes!!!
        pool_R.alloc_all(vec_ops_R, 1);
        pool_C.alloc_all(vec_ops_C, 1);
        pool_BR.alloc_all(vec_ops_R, 5);
        pool_BC.alloc_all(vec_ops_C, 6);        



    }


    //advection operator in classical form as F := ((V, \nabla) F).
    //V, F and ret are block vectors.
    void B_V_nabla_F(const BC_vec& Vel, const BC_vec& Func, BC_vec& ret)
    {
        BC_vec* CdFx = pool_BC.take();
        BC_vec* CdFy = pool_BC.take();
        BC_vec* CdFz = pool_BC.take();
        
        BR_vec* RdFx = pool_BR.take();
        BR_vec* RdFy = pool_BR.take();
        BR_vec* RdFz = pool_BR.take();
        BR_vec* VelR = pool_BR.take();

        kern->apply_grad3(Func.x, Func.y, Func.z, mask23, grad_x, grad_y, grad_z, CdFx->x, CdFx->y, CdFx->z, CdFy->x, CdFy->y, CdFy->z, CdFz->x, CdFz->y, CdFz->z);

        ifft(*CdFx, *RdFx);
        ifft(*CdFy, *RdFy);
        ifft(*CdFz, *RdFz);
        ifft(Vel, *VelR);
        
        pool_BC.release(CdFx);
        pool_BC.release(CdFy);
        pool_BC.release(CdFz);

        BR_vec* resR = pool_BR.take();
        kern->multiply_advection(VelR->x, VelR->y, VelR->z, RdFx->x, RdFx->y, RdFx->z, RdFy->x, RdFy->y, RdFy->z, RdFz->x, RdFz->y, RdFz->z, resR->x, resR->y, resR->z);

        fft(*resR, ret);
        // imag_vec(ret); //make sure it's pure imag after approximate advection!
        
        pool_BR.release(resR);
        pool_BR.release(RdFx);
        pool_BR.release(RdFy);
        pool_BR.release(RdFz);
        pool_BR.release(VelR);

    }

    void init_all_derivatives()
    {
        set_gradient_coefficients_low_level();
        set_Laplace_coefficients();
        set_force();
        kern->apply_mask(mask23);
    }


    void smooth(TR tau, BC_vec& U_in_place)
    {
        kern->apply_smooth(tau, Laplace, U_in_place.x, U_in_place.y, U_in_place.z);
    }

    void grad(const TC_vec& u_in, BC_vec& U_out)
    {
        kern->apply_grad(u_in, grad_x, grad_y, grad_z, U_out.x, U_out.y, U_out.z);
    }

    void div(const BC_vec& U_in, TC_vec& u_out)
    {
        kern->apply_div(U_in.x, U_in.y, U_in.z, grad_x, grad_y, grad_z, u_out);
    }

    //projects fourier vector to the divergence free space
    void project(BC_vec& U_in_place)
    {
        
        C_vec* uC0 = pool_C.take();
        BC_vec* UC1 = pool_BC.take();

        div(U_in_place, uC0->x); 
        kern->apply_iLaplace(Laplace, uC0->x);
        grad(uC0->x, *UC1);
        kern->add_mul3(TC(-1.0,0), UC1->x, UC1->y, UC1->z, U_in_place.x, U_in_place.y, U_in_place.z);
        
        pool_C.release(uC0);
        pool_BC.release(UC1);

    }

    // void imag_vec(BC_vec& U_in_place)
    // {
    //     kern->imag_vector(U_in_place.x, U_in_place.y, U_in_place.z);
    // }

    void set_gradient_coefficients_low_level()
    {
        TC_vec k_nabla = (TC_vec) malloc(Nx*sizeof(TC)); 
        for(int j=0;j<Nx; j++)
        {
            int m=j;
            if(j>=int(Nx/2))
                m=j-Nx;
            k_nabla[j]=TC(0, TR(m));
        }
        host_2_device_cpy<TC>(grad_x, k_nabla, Nx);
        free(k_nabla);

        k_nabla = (TC_vec) malloc(Ny*sizeof(TC));        
        for(int k=0;k<Ny; k++){
            int n=k;
            if(k>=int(Ny/2))
                n=k-Ny;
            k_nabla[k]=TC(0, n);
        }
        host_2_device_cpy<TC>(grad_y, k_nabla, Ny);
        free(k_nabla);        

        k_nabla = (TC_vec) malloc(Mz*sizeof(TC));                
        for(int l=0;l<Mz; l++){
            int q=l;
            //if(l>=Nz/2)  Due to reality condition
            //  q=l-Nz;
            k_nabla[l]=TC(0, q);
        }
        host_2_device_cpy<TC>(grad_z, k_nabla, Mz);
        free(k_nabla);        

    }

    void set_Laplace_coefficients()
    {
        kern->Laplace_Fourier(grad_x, grad_y, grad_z, Laplace);
    }
   
    void set_force()
    {
        kern->force_ABC(forceABC_R.x, forceABC_R.y, forceABC_R.z);
        fft(forceABC_R, forceABC);
    }

    void V2C(const T_vec& u_in, BC_vec& U_out)
    {
        kern->vec2complex(u_in, U_out.x, U_out.y, U_out.z);
    }
    void C2V(const BC_vec& U_in, T_vec& u_out)
    {
        kern->complex2vec(U_in.x, U_in.y, U_in.z, u_out);
    }

    TR norm(const BC_vec& U_in)
    {
        T norm_x = vec_ops_C->norm_l2(U_in.x);
        T norm_y = vec_ops_C->norm_l2(U_in.y);
        T norm_z = vec_ops_C->norm_l2(U_in.z);
        return(std::sqrt(norm_x*norm_x+norm_y*norm_y+norm_z*norm_z));
    }

    void ifft(const BC_vec& U_hat_, BR_vec& U)
    {
        ifft(U_hat_.x, U.x);
        ifft(U_hat_.y, U.y);
        ifft(U_hat_.z, U.z);
    }

    void fft(const BR_vec& U, BC_vec& U_hat)
    {
        fft(U.x, U_hat.x);
        fft(U.y, U_hat.y);
        fft(U.z, U_hat.z);
    }
    //use '1' or '2'
    void ifft(const TC_vec& u_hat_, TR_vec& u_)
    {
        FFT->ifft(u_hat_, u_);
        if(abc_flow_fft_rescale == 1)
        {
            T scale = T(1.0)/(Nx*Ny*Nz); //1
            vec_ops_R->scale(scale, u_); //1
        }
        if(abc_flow_fft_rescale == 2)
        {
            T scale = T(1.0)/std::sqrt(static_cast<T>(Nx*Ny*Nz)); //1
            vec_ops_R->scale(scale, u_); //1
        }
    }
    void fft(const TR_vec& u_, TC_vec& u_hat_)
    {
        
        FFT->fft(u_, u_hat_);
        if(abc_flow_fft_rescale == 0)
        {
            T scale = T(1.0)/(Nx*Ny*Nz); 
            vec_ops_C->scale(scale, u_hat_);
        }
        if(abc_flow_fft_rescale == 2)
        {
            T scale = T(1.0)/std::sqrt(static_cast<T>(Nx*Ny*Nz)); 
            vec_ops_C->scale(scale, u_hat_);
        }
    }


    std::tuple<TR,TR,TR> shift_phases(BC_vec& U_in)
    {
        auto varphis = kern->get_shift_phases(U_in.z); //test
        return varphis;
    }
    
    void translate_solution(BC_vec& U_in, TR varphi_x, TR varphi_y, TR varphi_z, BC_vec& U_out)
    {

        kern->apply_translate(U_in.x, grad_x, grad_y, grad_z, varphi_x, varphi_y, varphi_z, U_out.x);
        kern->apply_translate(U_in.y, grad_x, grad_y, grad_z, varphi_x, varphi_y, varphi_z, U_out.y);
        kern->apply_translate(U_in.z, grad_x, grad_y, grad_z, varphi_x, varphi_y, varphi_z, U_out.z);
        hermitian_symmetry(U_out, U_out);
        
    }

    void hermitian_symmetry(BC_vec& U, BC_vec& U_sym)
    {
        kern->make_hermitian_symmetric(U.x, U_sym.x);
        kern->make_hermitian_symmetric(U.y, U_sym.y);
        kern->make_hermitian_symmetric(U.z, U_sym.z);
    }


};

}

#endif
