#ifndef __KOLMOGOROV_2D_IMPL_CUH__
#define __KOLMOGOROV_2D_IMPL_CUH__


#include <common/macros.h>
#include <nonlinear_operators/Kolmogorov_flow_2D/Kolmogorov_2D_ker.h>


template<typename TC, typename TC_vec>
__global__ void Laplace_Fourier_kernel(size_t Nx, size_t Ny, TC_vec grad_x, TC_vec grad_y, TC_vec Laplace)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    TC x2 = grad_x[j]*grad_x[j];
    TC y2 = grad_y[k]*grad_y[k];    

    Laplace[I2(j,k,Ny)] = TC(x2.real() + y2.real(), 0.0);

    Laplace[0] = TC(1.0,0.0);


}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::Laplace_Fourier(TC_vec grad_x, TC_vec grad_y,  TC_vec Laplace)
{
    Laplace_Fourier_kernel<TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, grad_x, grad_y, Laplace);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_Laplace_kernel(size_t Nx, size_t Ny, T coeff, TC_vec Laplace, TC_vec ux, TC_vec uy)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    TC mult = TC(Laplace[I2(j,k,Ny)].real()*coeff,0);
    ux[I2(j,k,Ny)]*=mult;
    uy[I2(j,k,Ny)]*=mult;

    ux[0] = TC(0, 0);
    uy[0] = TC(0, 0);


}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_Laplace(TR coeff, TC_vec Laplace, TC_vec ux, TC_vec uy)
{
    apply_Laplace_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, coeff, Laplace, ux, uy);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void vec2complex_kernel(size_t N, T_vec v_in, TC_vec u_x, TC_vec u_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N-1) return; //N-1 due to zero in the begining
    
    u_x[0] = TC(0,0);
    u_y[0] = TC(0,0);

    u_x[i+1] = TC(0, v_in[i]);
    u_y[i+1] = TC(0, v_in[i+(N-1)]);
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::vec2complex(TR_vec v_in, TC_vec u_x, TC_vec u_y)
{

    vec2complex_kernel<TR, TR_vec, TC, TC_vec><<<dimGrid1C, dimBlock1>>>(NC, v_in, u_x, u_y);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_Fourier_kernel(int n, size_t Nx, size_t Ny, size_t scale, TC_vec force_x, TC_vec force_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    force_x[I2(j,k,Ny)] = TC(0,0);
    force_y[I2(j,k,Ny)] = TC(0,0);

    force_x[I2(0, n, Ny)]=TC(0, T(scale) );
    //force_x[I2(0,Ny-n, Ny)]=TC(0, T(0.5*scale) );

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::force_Fourier(int n, TC_vec force_x, TC_vec force_y)
{
    force_Fourier_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(n, Nx, My, Nx*Ny, force_x, force_y);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_ABC_kernel(size_t Nx, size_t Ny, T_vec force_x, T_vec force_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    T x = T(j)*T(4.0*M_PI)/T(Nx);
    T y = T(k)*T(2.0*M_PI)/T(Ny);

    force_x[I2(j,k,Ny)] = sin(y);
    force_y[I2(j,k,Ny)] = sin(x);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::force_ABC(TR_vec force_x, TR_vec force_y)
{
    force_ABC_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, force_x, force_y);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void complex2vec_kernel(size_t N, TC_vec u_x, TC_vec u_y, T_vec v_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N-1) return; //N-1 due to zero in the begining
    
    v_out[i] = T(u_x[i+1].imag());
    v_out[i+(N-1)] = T(u_y[i+1].imag());

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::complex2vec(TC_vec u_x, TC_vec u_y, TR_vec v_out)
{

    complex2vec_kernel<TR, TR_vec, TC, TC_vec><<<dimGrid1C, dimBlock1>>>(NC, u_x, u_y, v_out);

}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_grad_kernel(size_t Nx, size_t Ny,  TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec v_x, TC_vec v_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    v_x[I2(j,k,Ny)] = grad_x[j]*v_in[I2(j,k,Ny)];
    v_y[I2(j,k,Ny)] = grad_y[k]*v_in[I2(j,k,Ny)];

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_grad(TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec v_x, TC_vec v_y)
{

    apply_grad_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, v_in, grad_x, grad_y, v_x, v_y);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_div_kernel(size_t Nx, size_t Ny, TC_vec u_x, TC_vec u_y, TC_vec grad_x, TC_vec grad_y, TC_vec v_out)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    v_out[I2(j,k,Ny)] = grad_x[j]*u_x[I2(j,k,Ny)] + grad_y[k]*u_y[I2(j,k,Ny)];

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_div(TC_vec u_x, TC_vec u_y, TC_vec grad_x, TC_vec grad_y, TC_vec v_out)
{

    apply_div_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, u_x, u_y, grad_x, grad_y, v_out);

}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_projection_kernel(size_t Nx, size_t Ny, TC_vec u_x, TC_vec u_y, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec v_x, TC_vec v_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    TC ddd = grad_x[j]*u_x[I2(j,k,Ny)] + grad_y[k]*u_y[I2(j,k,Ny)];

    T lap = Laplace[I2(j,k,Ny)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I2(j,k,Ny)] = u_x[I2(j,k,Ny)] - grad_x[j]*ddd/lap;
    v_y[I2(j,k,Ny)] = u_y[I2(j,k,Ny)] - grad_y[k]*ddd/lap;

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_projection(TC_vec u_x, TC_vec u_y, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec v_x, TC_vec v_y)
{

    apply_projection_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, u_x, u_y, Laplace, grad_x, grad_y, v_x, v_y);

}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_smooth_kernel(size_t Nx, size_t Ny, T tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    T il = tau*Laplace[I2(j,k,Ny)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I2(j,k,Ny)]/=(T(1.0)-il);
    v_y[I2(j,k,Ny)]/=(T(1.0)-il);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_smooth(TR tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y)
{

    apply_smooth_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, tau, Laplace, v_x, v_y);

}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void imag_vector_kernel(size_t Nx, size_t Ny, TC_vec v_x, TC_vec v_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    

    v_x[I2(j,k,Ny)] = TC(0.0, v_x[I2(j,k,Ny)].imag());
    v_y[I2(j,k,Ny)] = TC(0.0, v_y[I2(j,k,Ny)].imag());

    v_x[0] = TC(0,0);
    v_y[0] = TC(0,0);
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::imag_vector(TC_vec v_x, TC_vec v_y)
{

    imag_vector_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, v_x, v_y);

}

template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace_kernel(size_t Nx, size_t Ny, TC_vec Laplace,  TC_vec v, T coeff)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    

    T lap = Laplace[I2(j,k,Ny)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v[I2(j,k,Ny)]*=TC(coeff/lap,0);
    v[0] = TC(0,0);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace(TC_vec Laplace, TC_vec v, TR coeff)
{

    apply_iLaplace_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, Laplace, v, coeff);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace3_kernel(size_t Nx, size_t Ny, TC_vec Laplace,  TC_vec v_x, TC_vec v_y, T coeff)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    

    T lap = Laplace[I2(j,k,Ny)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I2(j,k,Ny)]*=coeff/lap;
    v_y[I2(j,k,Ny)]*=coeff/lap;

    v_x[0] = TC(0,0);
    v_y[0] = TC(0,0);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TR coeff)
{

    apply_iLaplace3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, Laplace, v_x, v_y, coeff);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_abs_kernel(size_t Nx, size_t Ny, T_vec ux, T_vec uy, T_vec v)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    v[I2(j,k,Ny)] = sqrt(ux[I2(j,k,Ny)]*ux[I2(j,k,Ny)] + uy[I2(j,k,Ny)]*uy[I2(j,k,Ny)]);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_abs(TR_vec ux, TR_vec uy, TR_vec v)
{

    apply_abs_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, ux, uy, v);
}



template<typename TC, typename TC_vec>
__global__ void add_mul3_kernel(size_t N, TC alpha, TC_vec u_x, TC_vec u_y, TC_vec v_x, TC_vec v_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    v_x[i]+=alpha*u_x[i];
    v_y[i]+=alpha*u_y[i];   

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TC alpha, TC_vec u_x, TC_vec u_y, TC_vec v_x, TC_vec v_y)
{

    add_mul3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(Nx*My, alpha, u_x, u_y, v_x, v_y);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TR alpha, TR_vec u_x, TR_vec u_y, TR_vec v_x, TR_vec v_y)
{

    add_mul3_kernel<TR, TR_vec><<<dimGrid1R, dimBlock1>>>(Nx*Ny, alpha, u_x, u_y, v_x, v_y);

}


template<typename T,typename T_vec, typename TC, typename TC_vec>
__global__ void apply_mask_kernel(T alpha, size_t Nx, size_t Ny, TC_vec mask_2_3)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    int m,n;
    m=j; //X-coord
    if(j>=Nx/2)
        m=j-Nx;    
    n=k; //Y-coord    
    

    T kx=alpha*T(m);
    T ky=T(n);
    T kxMax=alpha*T(Nx);
    T kyMax=T(Ny);

    T sphere2=(kx*kx/(kxMax*kxMax)+ky*ky/(kyMax*kyMax));

    TC mask_val = TC((sphere2<=T(4.0/9.0)?(T(1)):(T(0))),0);
    mask_2_3[I2(j,k,Ny)] = mask_val;
    mask_2_3[0]=TC(0,0);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_mask(TR alpha, TC_vec mask_2_3)
{

    apply_mask_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(alpha, Nx, My, mask_2_3);

}





template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_grad3_kernel(size_t Nx, size_t Ny, TC_vec u1, TC_vec u2, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec v1_x, TC_vec v1_y, TC_vec v2_x, TC_vec v2_y,  TC_vec v3_x, TC_vec v3_y)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    TC gx = grad_x[j];
    TC gy = grad_y[k];
    TC u1l = u1[I2(j,k,Ny)];
    TC u2l = u2[I2(j,k,Ny)];
    TC mask_l = mask[I2(j,k,Ny)];

    v1_x[I2(j,k,Ny)] = gx*u1l*mask_l;
    v1_y[I2(j,k,Ny)] = gy*u1l*mask_l;
    
    v2_x[I2(j,k,Ny)] = gx*u2l*mask_l;
    v2_y[I2(j,k,Ny)] = gy*u2l*mask_l;

    v1_x[0] = TC(0,0);
    v1_y[0] = TC(0,0);
    
    v2_x[0] = TC(0,0);
    v2_y[0] = TC(0,0);
    
    v3_x[0] = TC(0,0);
    v3_y[0] = TC(0,0);

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::apply_grad3(TC_vec u1, TC_vec u2, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec v1_x, TC_vec v1_y, TC_vec v2_x, TC_vec v2_y, TC_vec v3_x, TC_vec v3_y)
{

    apply_grad3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, My, u1, u2, mask, grad_x, grad_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void multiply_advection_kernel(size_t Nx, size_t Ny, T_vec Vx, T_vec Vy, T_vec Fx_x, T_vec Fx_y, T_vec Fy_x, T_vec Fy_y, T_vec Fz_x, T_vec Fz_y, T_vec resx, T_vec resy)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;

    T vx = Vx[I2(j,k,Ny)];
    T vy = Vy[I2(j,k,Ny)];

    resx[I2(j,k,Ny)] = vx*Fx_x[I2(j,k,Ny)] + vy*Fx_y[I2(j,k,Ny)];

    resy[I2(j,k,Ny)] = vx*Fy_x[I2(j,k,Ny)] + vy*Fy_y[I2(j,k,Ny)];
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::multiply_advection(TR_vec Vx, TR_vec Vy, TR_vec Fx_x, TR_vec Fx_y, TR_vec Fy_x, TR_vec Fy_y, TR_vec Fz_x, TR_vec Fz_y, TR_vec resx, TR_vec resy)
{
    multiply_advection_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, Vx, Vy, Fx_x, Fx_y, Fy_x, Fy_y, Fz_x, Fz_y, resx, resy);
}



template<typename TC, typename TC_vec>
__global__ void negate3_kernel(size_t N, TC_vec v_x, TC_vec v_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    TC vx = v_x[i];
    v_x[i] = -vx;

    TC vy = v_y[i];
    v_y[i] = -vy;

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::negate3(TC_vec v_x, TC_vec v_y)
{

    negate3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(Nx*My, v_x, v_y);

}


template<typename TC, typename TC_vec>
__global__ void copy3_kernel(TC_vec u_x, TC_vec u_y, size_t N, TC_vec v_x, TC_vec v_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    TC vx = u_x[i];
    v_x[i] = vx;

    TC vy = u_y[i];
    v_y[i] = vy;

}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::copy3(TC_vec u_x, TC_vec u_y, TC_vec v_x, TC_vec v_y)
{

    copy3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(u_x, u_y, Nx*My, v_x, v_y);

}



template<typename T, typename T_vec>
__global__ void B_ABC_exact_kernel(size_t Nx, size_t Ny, T coeff, T_vec ux, T_vec uy)
{
int j=blockDim.x * blockIdx.x + threadIdx.x;
int k=blockDim.y * blockIdx.y + threadIdx.y;
    
if((j>=Nx)||(k>=Ny)) return;
    
    T x = T(j)*T(4.0*M_PI)/T(Nx);
    T y = T(k)*T(2.0*M_PI)/T(Ny);

    T sx = coeff*sin(x);
    T sy = coeff*sin(y);

    T cx = coeff*cos(x);
    T cy = coeff*cos(y);

    ux[I2(j,k,Ny)] = sx*cy;
    uy[I2(j,k,Ny)] = cx*sy;
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_2D_ker<TR, TR_vec, TC, TC_vec>::B_ABC_exact(TR coeff, TR_vec ux, TR_vec uy)
{

    B_ABC_exact_kernel<TR, TR_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, coeff, ux, uy);
}



#endif // __KOLMOGOROV_2D_IMPL_CUH__