#ifndef __Taylor_Green_IMPL_CUH__
#define __Taylor_Green_IMPL_CUH__


#include <common/macros.h>
#include <nonlinear_operators/Taylor_Green/Taylor_Green_ker.h>





template<typename TC, typename TC_vec>
__global__ void apply_curl_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec ux_in, TC_vec uy_in, TC_vec uz_in, TC_vec wx_out, TC_vec wy_out, TC_vec wz_out)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:


    wx_out[I3(j,k,l)] = grad_y[k]*uz_in[I3(j,k,l)] - grad_z[l]*uy_in[I3(j,k,l)];
    wy_out[I3(j,k,l)] = -(grad_x[j]*uz_in[I3(j,k,l)] - grad_z[l]*ux_in[I3(j,k,l)]);
    wz_out[I3(j,k,l)] = grad_x[j]*uy_in[I3(j,k,l)] - grad_y[k]*ux_in[I3(j,k,l)];

}
}


template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_curl(TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec ux_in, TC_vec uy_in, TC_vec uz_in, TC_vec wx_out, TC_vec wy_out, TC_vec wz_out)
{
    apply_curl_kernel<TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, grad_x, grad_y, grad_z, ux_in,  uy_in, uz_in, wx_out, wy_out, wz_out);
}



template<typename TC, typename TC_vec>
__global__ void Laplace_Fourier_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Laplace)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    TC x2 = grad_x[j]*grad_x[j];
    TC y2 = grad_y[k]*grad_y[k];
    TC z2 = grad_z[l]*grad_z[l];    

    Laplace[I3(j,k,l)] = TC(x2.real() + y2.real() + z2.real(), 0.0);

    Laplace[0] = TC(1.0,0.0);

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::Laplace_Fourier(TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Laplace)
{
    Laplace_Fourier_kernel<TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, grad_x, grad_y, grad_z, Laplace);
}


template<typename TC, typename TC_vec>
__global__ void Biharmonic_Fourier_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Biharmonic)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    TC x2 = grad_x[j]*grad_x[j];
    TC y2 = grad_y[k]*grad_y[k];
    TC z2 = grad_z[l]*grad_z[l];

    TC x4 = x2*x2;
    TC y4 = y2*y2;
    TC z4 = z2*z2;    

    Biharmonic[I3(j,k,l)] = -TC(x4.real() + y4.real() + z4.real() + 2.0*(x2.real()*y2.real()+y2.real()*z2.real()+x2.real()*z2.real()), 0.0);

    Biharmonic[0] = TC(1.0,0.0);

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::Biharmonic_Fourier(TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Biharmonic)
{
    Biharmonic_Fourier_kernel<TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, grad_x, grad_y, grad_z, Biharmonic);
}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_Laplace_kernel(size_t Nx, size_t Ny, size_t Nz, T coeff, TC_vec Laplace, TC_vec ux, TC_vec uy, TC_vec uz)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    TC mult = TC(Laplace[I3(j,k,l)].real()*coeff,0);
    ux[I3(j,k,l)]*=mult;
    uy[I3(j,k,l)]*=mult;
    uz[I3(j,k,l)]*=mult;

    ux[0] = TC(0,0);
    uy[0] = TC(0,0);
    uz[0] = TC(0,0);

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_Laplace(TR coeff, TC_vec Laplace, TC_vec ux, TC_vec uy, TC_vec uz)
{
    apply_Laplace_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, coeff, Laplace, ux, uy, uz);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_Laplace_Biharmonic_kernel(size_t Nx, size_t Ny, size_t Nz, T coeff, TC_vec Laplace, T coeff_bihrmonic, TC_vec Biharmonic, TC_vec ux, TC_vec uy, TC_vec uz)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    TC mult = TC(Laplace[I3(j,k,l)].real()*coeff + Biharmonic[I3(j,k,l)].real()*coeff_bihrmonic ,0);
    ux[I3(j,k,l)]*=mult;
    uy[I3(j,k,l)]*=mult;
    uz[I3(j,k,l)]*=mult;

    ux[0] = TC(0,0);
    uy[0] = TC(0,0);
    uz[0] = TC(0,0);

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_Laplace_Biharmonic(TR coeff, TC_vec Laplace, TR coeff_bihrmonic, TC_vec Biharmonic, TC_vec ux, TC_vec uy, TC_vec uz)
{
    apply_Laplace_Biharmonic_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, coeff, Laplace, coeff_bihrmonic, Biharmonic, ux, uy, uz);
}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void vec2complex_kernel(size_t N, T_vec v_in, TC_vec u_x, TC_vec u_y, TC_vec u_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N-1) return; //N-1 due to zero in the begining
    
    u_x[0] = TC(0,0);
    u_y[0] = TC(0,0);
    u_z[0] = TC(0,0);

    u_x[i+1] = TC(0, v_in[i]);
    u_y[i+1] = TC(0, v_in[i+(N-1)]);
    u_z[i+1] = TC(0, v_in[i+2*(N-1)]);    

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::vec2complex(TR_vec v_in, TC_vec u_x, TC_vec u_y, TC_vec u_z)
{

    vec2complex_kernel<TR, TR_vec, TC, TC_vec><<<dimGrid1C, dimBlock1>>>(NC, v_in, u_x, u_y, u_z);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_Fourier_sin_kernel(int n_y, int n_z, T scale_const, size_t Nx, size_t Ny, size_t Nz, size_t scale, TC_vec force_x, TC_vec force_y, TC_vec force_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    force_x[I3(j,k,l)] = TC(0,0);
    force_y[I3(j,k,l)] = TC(0,0);
    force_z[I3(j,k,l)] = TC(0,0);

    force_x[I3(0,n_y,n_z)]=TC(0, T(scale)*scale_const );
    force_x[I3(0,Ny-n_y,n_z)]=TC(0, -T(scale)*scale_const );

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::force_Fourier_sin(int n_y, int n_z, TR scale_const, TC_vec force_x, TC_vec force_y, TC_vec force_z)
{
    force_Fourier_sin_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(n_y, n_z, scale_const, Nx, Ny, Mz, Nx*Ny*Nz, force_x, force_y, force_z);
}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_Fourier_sin_cos_kernel(int n_y, int n_z, T scale_const, size_t Nx, size_t Ny, size_t Nz, size_t scale, TC_vec force_x, TC_vec force_y, TC_vec force_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    force_x[I3(j,k,l)] = TC(0,0);
    force_y[I3(j,k,l)] = TC(0,0);
    force_z[I3(j,k,l)] = TC(0,0);

    force_x[I3(0,n_y,n_z)]=TC(0, -T(scale)*scale_const );
    force_x[I3(0,Ny-n_y,n_z)]=TC(0, -T(scale)*scale_const );

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::force_Fourier_sin_cos(int n_y, int n_z, TR scale_const, TC_vec force_x, TC_vec force_y, TC_vec force_z)
{
    force_Fourier_sin_cos_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(n_y, n_z, scale_const, Nx, Ny, Mz, Nx*Ny*Nz, force_x, force_y, force_z);
}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_ABC_kernel(size_t Nx, size_t Ny, size_t Nz, T_vec force_x, T_vec force_y, T_vec force_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    T x = T(j)*T(4.0*M_PI)/T(Nx);
    T y = T(k)*T(2.0*M_PI)/T(Ny);
    T z = T(l)*T(2.0*M_PI)/T(Nz);

    force_x[I3(j,k,l)] = sin(y)*cos(z);
    force_y[I3(j,k,l)] = sin(z)*cos(x);
    force_z[I3(j,k,l)] = sin(x)*cos(y);

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::force_ABC(TR_vec force_x, TR_vec force_y, TR_vec force_z)
{
    force_ABC_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, Nz, force_x, force_y, force_z);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void complex2vec_kernel(size_t N, TC_vec u_x, TC_vec u_y, TC_vec u_z, T_vec v_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N-1) return; //N-1 due to zero in the begining
    
    v_out[i] = T(u_x[i+1].imag());
    v_out[i+(N-1)] = T(u_y[i+1].imag());
    v_out[i+2*(N-1)] = T(u_z[i+1].imag());

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::complex2vec(TC_vec u_x, TC_vec u_y, TC_vec u_z, TR_vec v_out)
{

    complex2vec_kernel<TR, TR_vec, TC, TC_vec><<<dimGrid1C, dimBlock1>>>(NC, u_x, u_y, u_z, v_out);

}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_grad_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    v_x[I3(j,k,l)] = grad_x[j]*v_in[I3(j,k,l)];
    v_y[I3(j,k,l)] = grad_y[k]*v_in[I3(j,k,l)];
    v_z[I3(j,k,l)] = grad_z[l]*v_in[I3(j,k,l)];

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_grad(TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    apply_grad_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, v_in, grad_x, grad_y, grad_z, v_x, v_y, v_z);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_div_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec u_x, TC_vec u_y, TC_vec u_z,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_out)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    v_out[I3(j,k,l)] = grad_x[j]*u_x[I3(j,k,l)] + grad_y[k]*u_y[I3(j,k,l)] + grad_z[l]*u_z[I3(j,k,l)];

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_div(TC_vec u_x, TC_vec u_y, TC_vec u_z,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_out)
{

    apply_div_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, u_x, u_y, u_z, grad_x, grad_y, grad_z, v_out);

}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_projection_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    TC ddd = grad_x[j]*u_x[I3(j,k,l)] + grad_y[k]*u_y[I3(j,k,l)] + grad_z[l]*u_z[I3(j,k,l)];

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I3(j,k,l)] = u_x[I3(j,k,l)] - grad_x[j]*ddd/lap;
    v_y[I3(j,k,l)] = u_y[I3(j,k,l)] - grad_y[k]*ddd/lap;
    v_z[I3(j,k,l)] = u_z[I3(j,k,l)] - grad_z[l]*ddd/lap;

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_projection(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    apply_projection_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, u_x, u_y, u_z, Laplace, grad_x, grad_y, grad_z, v_x, v_y, v_z);

}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_smooth_kernel(size_t Nx, size_t Ny, size_t Nz, T tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    T il = tau*Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I3(j,k,l)]/=(T(1.0)-il);
    v_y[I3(j,k,l)]/=(T(1.0)-il);
    v_z[I3(j,k,l)]/=(T(1.0)-il);

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_smooth(TR tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    apply_smooth_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, tau, Laplace, v_x, v_y, v_z);

}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void imag_vector_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    v_x[I3(j,k,l)] = TC(0.0, v_x[I3(j,k,l)].imag());
    v_y[I3(j,k,l)] = TC(0.0, v_y[I3(j,k,l)].imag());
    v_z[I3(j,k,l)] = TC(0.0, v_z[I3(j,k,l)].imag());

    v_x[0] = TC(0,0);
    v_y[0] = TC(0,0);
    v_z[0] = TC(0,0);
}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::imag_vector(TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    imag_vector_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, v_x, v_y, v_z);

}

template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec Laplace,  TC_vec v, T coeff)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v[I3(j,k,l)]*=TC(coeff/lap,0);
    v[0] = TC(0,0);



}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace(TC_vec Laplace, TC_vec v, TR coeff)
{

    apply_iLaplace_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, v, coeff);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace3_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec Laplace,  TC_vec v_x, TC_vec v_y, TC_vec v_z, T coeff)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I3(j,k,l)]*=coeff/lap;
    v_y[I3(j,k,l)]*=coeff/lap;
    v_z[I3(j,k,l)]*=coeff/lap;

    v_x[0] = TC(0,0);
    v_y[0] = TC(0,0);
    v_z[0] = TC(0,0);

}
}


template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff)
{

    apply_iLaplace3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, v_x, v_y, v_z, coeff);
}

template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace3_Biharmonic3_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec Laplace, TC_vec Biharmonic,  TC_vec v_x, TC_vec v_y, TC_vec v_z, T coeff, T coeff_bihrmonic)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!
    T biharm = Biharmonic[I3(j,k,l)].real();
    v_x[I3(j,k,l)] /= (lap/coeff+biharm*coeff_bihrmonic);
    v_y[I3(j,k,l)] /= (lap/coeff+biharm*coeff_bihrmonic);
    v_z[I3(j,k,l)] /= (lap/coeff+biharm*coeff_bihrmonic);

    v_x[0] = TC(0,0);
    v_y[0] = TC(0,0);
    v_z[0] = TC(0,0);

}
}


template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3_Biharmonic3(TC_vec Laplace, TC_vec Biharmonic,  TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff, TR coeff_bihrmonic)
{

    apply_iLaplace3_Biharmonic3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, Biharmonic, v_x, v_y, v_z, coeff, coeff_bihrmonic);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace3_plus_E_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec Laplace,  TC_vec v_x, TC_vec v_y, TC_vec v_z, T coeff, T a, T b)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!

    v_x[I3(j,k,l)]/=(a+b*lap/coeff);
    v_y[I3(j,k,l)]/=(a+b*lap/coeff);
    v_z[I3(j,k,l)]/=(a+b*lap/coeff);

    // v_x[0] = 0.0;
    // v_y[0] = 0.0;
    // v_z[0] = 0.0;

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3_plus_E(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff, TR a, TR b)
{

    apply_iLaplace3_plus_E_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, v_x, v_y, v_z, coeff, a, b);
}

template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_iLaplace3_Biharmonic3_plus_E_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec Laplace, TC_vec Biharmonic,  TC_vec v_x, TC_vec v_y, TC_vec v_z, T coeff, T coeff_bihrmonic, T a, T b)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    

    T lap = Laplace[I3(j,k,l)].real(); //Laplace is assumed to have 1 at the zero wavenumber!
    T biharm = Biharmonic[I3(j,k,l)].real();
    T opt = (lap/coeff+biharm*coeff_bihrmonic);

    v_x[I3(j,k,l)]/=(a+b*opt);
    v_y[I3(j,k,l)]/=(a+b*opt);
    v_z[I3(j,k,l)]/=(a+b*opt);

    // v_x[0] = TC(0,0);
    // v_y[0] = TC(0,0);
    // v_z[0] = TC(0,0);

}
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3_Biharmonic3_plus_E(TC_vec Laplace, TC_vec Biharmonic, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff, TR coeff_bihrmonic, TR a, TR b)
{

    apply_iLaplace3_Biharmonic3_plus_E_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, Biharmonic, v_x, v_y, v_z, coeff, coeff_bihrmonic, a, b);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_abs_kernel(size_t N, T_vec ux, T_vec uy, T_vec uz, T_vec v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    v[i] = sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_abs(TR_vec ux, TR_vec uy, TR_vec uz, TR_vec v)
{
    size_t NNX = Nx*Ny*Nz;
    dim3 blocks_x=(NNX+BLOCKSIZE)/BLOCKSIZE;
    apply_abs_kernel<TR, TR_vec, TC, TC_vec><<<blocks_x, dimBlock1>>>(NNX, ux, uy, uz, v);
}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_abs(size_t Nx_l, size_t Ny_l, size_t Nz_l, TR_vec ux, TR_vec uy, TR_vec uz, TR_vec v)
{
    size_t NRR = Nx_l*Ny_l*Nz_l;
    dim3 blocks_x=(NRR+BLOCKSIZE)/BLOCKSIZE;
    apply_abs_kernel<TR, TR_vec, TC, TC_vec><<<blocks_x, dimBlock1>>>(NRR, ux, uy, uz, v);
}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_scale_inplace_kernel(size_t N, T scale, T_vec ux, T_vec uy, T_vec uz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    ux[i]*=scale;
    uy[i]*=scale;
    uz[i]*=scale;

}


template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_scale_inplace(size_t Nx_l, size_t Ny_l, size_t Nz_l, TR scale, TR_vec ux, TR_vec uy, TR_vec uz)
{
    size_t NRR = Nx_l*Ny_l*Nz_l;
    dim3 blocks_x=(NRR+BLOCKSIZE)/BLOCKSIZE;
    apply_scale_inplace_kernel<TR, TR_vec, TC, TC_vec><<<blocks_x, dimBlock1>>>(NRR, scale, ux, uy, uz);
}


template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void convert_size_kernel(size_t Nx_src, size_t Ny_src, size_t Nz_src, size_t Nx_dest, size_t Ny_dest, size_t Nz_dest, T scale, TC_vec ux_src_hat, TC_vec uy_src_hat, TC_vec uz_src_hat, TC_vec ux_dest_hat, TC_vec uy_dest_hat, TC_vec uz_dest_hat)
{
size_t Nx = Nx_src; size_t Ny = Ny_src; size_t Nz = Nz_src;
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    int jj=j;
    
    if(j>=Nx_src/2)
        jj=j+(Nx_dest-Nx_src);
    int kk=k;
    if(k>=Ny_src/2)
        kk=k+(Ny_dest-Ny_src);

    ux_dest_hat[_I3(jj,kk,l,Nx_dest, Ny_dest, Nz_dest)]=scale*ux_src_hat[_I3(j,k,l,Nx_src, Ny_src, Nz_src)];
    uy_dest_hat[_I3(jj,kk,l,Nx_dest, Ny_dest, Nz_dest)]=scale*uy_src_hat[_I3(j,k,l,Nx_src, Ny_src, Nz_src)];
    uz_dest_hat[_I3(jj,kk,l,Nx_dest, Ny_dest, Nz_dest)]=scale*uz_src_hat[_I3(j,k,l,Nx_src, Ny_src, Nz_src)];

}
}


template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::convert_size(size_t Nx_dest, size_t Ny_dest, size_t Mz_dest, TR scale, TC_vec ux_src_hat, TC_vec uy_src_hat, TC_vec uz_src_hat, TC_vec ux_dest_hat, TC_vec uy_dest_hat, TC_vec uz_dest_hat)
{
    convert_size_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Nx_dest, Ny_dest, Mz_dest, scale, ux_src_hat, uy_src_hat, uz_src_hat, ux_dest_hat, uy_dest_hat, uz_dest_hat);
}



template<typename TC, typename TC_vec>
__global__ void add_mul3_kernel(size_t N, TC alpha, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    v_x[i]+=alpha*u_x[i];
    v_y[i]+=alpha*u_y[i];
    v_z[i]+=alpha*u_z[i];

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TC alpha, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    add_mul3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(Nx*Ny*Mz, alpha, u_x, u_y, u_z, v_x, v_y, v_z);

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TR alpha, TR_vec u_x, TR_vec u_y, TR_vec u_z, TR_vec v_x, TR_vec v_y, TR_vec v_z)
{

    add_mul3_kernel<TR, TR_vec><<<dimGrid1R, dimBlock1>>>(Nx*Ny*Nz, alpha, u_x, u_y, u_z, v_x, v_y, v_z);

}


template<typename T,typename T_vec, typename TC, typename TC_vec>
__global__ void apply_mask_kernel(T alpha, size_t Nx, size_t Ny, size_t Nz, TC_vec mask_2_3)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    int q,m,n;
    q=l; //Z-coord
    m=j; //Y-coord
    if(j>=Nx/2)
        m=j-Nx;    
    n=k; //X-coord
    if(k>=Ny/2)
        n=k-Ny;     
    

    T kx=alpha*T(m);
    T ky=T(n);
    T kz=T(q);
    T kxMax=alpha*T(Nx);
    T kyMax=T(Ny);
    T kzMax=T(Nz);

    T sphere2=(kx*kx/(kxMax*kxMax)+ky*ky/(kyMax*kyMax)+kz*kz/(kzMax*kzMax));

    TC mask_val = TC((sphere2<=T(4.0/9.0)?(T(1)):(T(0))),0);
    mask_2_3[I3(j,k,l)] = mask_val;
    mask_2_3[0]=TC(0,0);

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_mask(TR alpha, TC_vec mask_2_3)
{

    apply_mask_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(alpha, Nx, Ny, Mz, mask_2_3);

}





template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_grad3_kernel(size_t Nx, size_t Ny, size_t Nz, TC_vec u1, TC_vec u2, TC_vec u3, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v1_x, TC_vec v1_y, TC_vec v1_z, TC_vec v2_x, TC_vec v2_y, TC_vec v2_z, TC_vec v3_x, TC_vec v3_y, TC_vec v3_z)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    TC gx = grad_x[j];
    TC gy = grad_y[k];
    TC gz = grad_z[l];
    TC u1l = u1[I3(j,k,l)];
    TC u2l = u2[I3(j,k,l)];
    TC u3l = u3[I3(j,k,l)];
    TC mask_l = mask[I3(j,k,l)];
    //TC gx = 1, gy = 1, gz = 1, mask_l = 1;

    v1_x[I3(j,k,l)] = gx*u1l*mask_l;
    v1_y[I3(j,k,l)] = gy*u1l*mask_l;
    v1_z[I3(j,k,l)] = gz*u1l*mask_l;
    
    v2_x[I3(j,k,l)] = gx*u2l*mask_l;
    v2_y[I3(j,k,l)] = gy*u2l*mask_l;
    v2_z[I3(j,k,l)] = gz*u2l*mask_l;
    
    v3_x[I3(j,k,l)] = gx*u3l*mask_l;
    v3_y[I3(j,k,l)] = gy*u3l*mask_l;
    v3_z[I3(j,k,l)] = gz*u3l*mask_l;

    v1_x[0] = TC(0,0);
    v1_y[0] = TC(0,0);
    v1_z[0] = TC(0,0);
    
    v2_x[0] = TC(0,0);
    v2_y[0] = TC(0,0);
    v2_z[0] = TC(0,0);
    
    v3_x[0] = TC(0,0);
    v3_y[0] = TC(0,0);
    v3_z[0] = TC(0,0);

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::apply_grad3(TC_vec u1, TC_vec u2, TC_vec u3, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v1_x, TC_vec v1_y, TC_vec v1_z, TC_vec v2_x, TC_vec v2_y, TC_vec v2_z, TC_vec v3_x, TC_vec v3_y, TC_vec v3_z)
{

    apply_grad3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, u1, u2, u3, mask, grad_x, grad_y, grad_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z);

}





template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void multiply_advection_kernel(size_t Nx, size_t Ny, size_t Nz, T_vec Vx, T_vec Vy, T_vec Vz, T_vec Fx_x, T_vec Fx_y, T_vec Fx_z, T_vec Fy_x, T_vec Fy_y, T_vec Fy_z, T_vec Fz_x, T_vec Fz_y, T_vec Fz_z, T_vec resx, T_vec resy, T_vec resz)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:

    T vx = Vx[I3(j,k,l)];
    T vy = Vy[I3(j,k,l)];
    T vz = Vz[I3(j,k,l)];

    resx[I3(j,k,l)] = vx*Fx_x[I3(j,k,l)] + vy*Fx_y[I3(j,k,l)] + vz*Fx_z[I3(j,k,l)];

    resy[I3(j,k,l)] = vx*Fy_x[I3(j,k,l)] + vy*Fy_y[I3(j,k,l)] + vz*Fy_z[I3(j,k,l)];

    resz[I3(j,k,l)] = vx*Fz_x[I3(j,k,l)] + vy*Fz_y[I3(j,k,l)] + vz*Fz_z[I3(j,k,l)];    

    

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::multiply_advection(TR_vec Vx, TR_vec Vy, TR_vec Vz, TR_vec Fx_x, TR_vec Fx_y, TR_vec Fx_z, TR_vec Fy_x, TR_vec Fy_y, TR_vec Fy_z, TR_vec Fz_x, TR_vec Fz_y, TR_vec Fz_z, TR_vec resx, TR_vec resy, TR_vec resz)
{
    multiply_advection_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, Nz, Vx, Vy, Vz, Fx_x, Fx_y, Fx_z, Fy_x, Fy_y, Fy_z, Fz_x, Fz_y, Fz_z, resx, resy, resz);
}



template<typename TC, typename TC_vec>
__global__ void negate3_kernel(size_t N, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    TC vx = v_x[i];
    v_x[i] = -vx;

    TC vy = v_y[i];
    v_y[i] = -vy;
    
    TC vz = v_z[i];
    v_z[i] = -vz;


}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::negate3(TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    negate3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(Nx*Ny*Mz, v_x, v_y, v_z);

}


template<typename TC, typename TC_vec>
__global__ void copy3_kernel(TC_vec u_x, TC_vec u_y, TC_vec u_z, size_t N, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    
    TC vx = u_x[i];
    v_x[i] = vx;

    TC vy = u_y[i];
    v_y[i] = vy;
    
    TC vz = u_z[i];
    v_z[i] = vz;


}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::copy3(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    copy3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(u_x, u_y, u_z, Nx*Ny*Mz, v_x, v_y, v_z);

}


template<typename TC, typename TC_vec>
__global__ void copy_mul_poinwise_3_kernel(TC_vec mask, TC_vec u_x, TC_vec u_y, TC_vec u_z, size_t N, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    auto mask_l = mask[i];
    TC vx = u_x[i];
    v_x[i] = vx*mask_l;

    TC vy = u_y[i];
    v_y[i] = vy*mask_l;
    
    TC vz = u_z[i];
    v_z[i] = vz*mask_l;


}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::copy_mul_poinwise_3(TC_vec mask, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    copy_mul_poinwise_3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(mask, u_x, u_y, u_z, Nx*Ny*Mz, v_x, v_y, v_z);

}

template<typename T, typename T_vec>
__global__ void B_ABC_exact_kernel(size_t Nx, size_t Ny, size_t Nz, T coeff, T_vec ux, T_vec uy, T_vec uz)
{
unsigned int t1, xIndex, yIndex, zIndex, index_in, gridIndex;
unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
index_in = ( gridIndex * blockDim.y + threadIdx.y )*blockDim.x + threadIdx.x;
if ( index_in < sizeOfData )
{
    t1 =  index_in/Nz; 
    zIndex = index_in - Nz*t1 ;
    xIndex =  t1/Ny; 
    yIndex = t1 - Ny * xIndex ;
    unsigned int j=xIndex, k=yIndex, l=zIndex;
//  operation starts from here:
    
    T x = T(j)*T(4.0*M_PI)/T(Nx);
    T y = T(k)*T(2.0*M_PI)/T(Ny);
    T z = T(l)*T(2.0*M_PI)/T(Nz);

    T sx = coeff*sin(x);
    T sy = coeff*sin(y);
    T sz = coeff*sin(z);
    T cx = coeff*cos(x);
    T cy = coeff*cos(y);
    T cz = coeff*cos(z);        

    ux[I3(j,k,l)] = sz*cx*cy*cz - sx*cy*sy*sz;
    uy[I3(j,k,l)] = sx*cy*cz*cx - sy*cz*sz*sx;
    uz[I3(j,k,l)] = sy*cy*cx*cz - sz*cx*sx*sy;

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Taylor_Green_ker<TR, TR_vec, TC, TC_vec>::B_ABC_exact(TR coeff, TR_vec ux, TR_vec uy, TR_vec uz)
{

    B_ABC_exact_kernel<TR, TR_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, Nz, coeff, ux, uy, uz);
}






#endif // __Taylor_Green_IMPL_CUH__