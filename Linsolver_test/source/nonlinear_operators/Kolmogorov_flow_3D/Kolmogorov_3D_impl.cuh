#ifndef __KOLMOGOROV_3D_IMPL_CUH__
#define __KOLMOGOROV_3D_IMPL_CUH__


#include <common/macros.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D_ker.h>


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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::Laplace_Fourier(TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Laplace)
{
    Laplace_Fourier_kernel<TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, grad_x, grad_y, grad_z, Laplace);
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_Laplace(TR coeff, TC_vec Laplace, TC_vec ux, TC_vec uy, TC_vec uz)
{
    apply_Laplace_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, coeff, Laplace, ux, uy, uz);
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::vec2complex(TR_vec v_in, TC_vec u_x, TC_vec u_y, TC_vec u_z)
{

    vec2complex_kernel<TR, TR_vec, TC, TC_vec><<<dimGrid1C, dimBlock1>>>(NC, v_in, u_x, u_y, u_z);

}



template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void force_Fourier_kernel(int n, size_t Nx, size_t Ny, size_t Nz, size_t scale, TC_vec force_x, TC_vec force_y, TC_vec force_z)
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

    force_x[I3(0,n,n)]=TC(0, -T(scale)*0.25 );
    force_x[I3(0,Ny-n,n)]=TC(0, -T(scale)*0.25 );

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::force_Fourier(int n, TC_vec force_x, TC_vec force_y, TC_vec force_z)
{
    force_Fourier_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(n, Nx, Ny, Mz, Nx*Ny*Nz, force_x, force_y, force_z);
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::force_ABC(TR_vec force_x, TR_vec force_y, TR_vec force_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::complex2vec(TC_vec u_x, TC_vec u_y, TC_vec u_z, TR_vec v_out)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_grad(TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_div(TC_vec u_x, TC_vec u_y, TC_vec u_z,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_out)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_projection(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_smooth(TR tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::imag_vector(TC_vec v_x, TC_vec v_y, TC_vec v_z)
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

    v[I3(j,k,l)]*=coeff/lap;



}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace(TC_vec Laplace, TC_vec v, TR coeff)
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



}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_iLaplace3(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff)
{

    apply_iLaplace3_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNC, dimBlockN>>>(Nx, Ny, Mz, Laplace, v_x, v_y, v_z, coeff);
}




template<typename T, typename T_vec, typename TC, typename TC_vec>
__global__ void apply_abs_kernel(size_t Nx, size_t Ny, size_t Nz, T_vec ux, T_vec uy, T_vec uz, T_vec v)
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
    
    v[I3(j,k,l)] = sqrt(ux[I3(j,k,l)]*ux[I3(j,k,l)] + uy[I3(j,k,l)]*uy[I3(j,k,l)] + uz[I3(j,k,l)]*uz[I3(j,k,l)]);

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_abs(TR_vec ux, TR_vec uy, TR_vec uz, TR_vec v)
{

    apply_abs_kernel<TR, TR_vec, TC, TC_vec><<<dimGridNR, dimBlockN>>>(Nx, Ny, Nz, ux, uy, uz, v);
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TC alpha, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    add_mul3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(Nx*Ny*Mz, alpha, u_x, u_y, u_z, v_x, v_y, v_z);

}

template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::add_mul3(TR alpha, TR_vec u_x, TR_vec u_y, TR_vec u_z, TR_vec v_x, TR_vec v_y, TR_vec v_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_mask(TR alpha, TC_vec mask_2_3)
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

    v1_x[I3(j,k,l)] = gx*u1l*mask_l;
    v1_y[I3(j,k,l)] = gy*u1l*mask_l;
    v1_z[I3(j,k,l)] = gz*u1l*mask_l;
    
    v2_x[I3(j,k,l)] = gx*u2l*mask_l;
    v2_y[I3(j,k,l)] = gy*u2l*mask_l;
    v2_z[I3(j,k,l)] = gz*u2l*mask_l;
    
    v3_x[I3(j,k,l)] = gx*u3l*mask_l;
    v3_y[I3(j,k,l)] = gy*u3l*mask_l;
    v3_z[I3(j,k,l)] = gz*u3l*mask_l;

}
}
template <typename TR, typename TR_vec, typename TC, typename TC_vec>
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::apply_grad3(TC_vec u1, TC_vec u2, TC_vec u3, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v1_x, TC_vec v1_y, TC_vec v1_z, TC_vec v2_x, TC_vec v2_y, TC_vec v2_z, TC_vec v3_x, TC_vec v3_y, TC_vec v3_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::multiply_advection(TR_vec Vx, TR_vec Vy, TR_vec Vz, TR_vec Fx_x, TR_vec Fx_y, TR_vec Fx_z, TR_vec Fy_x, TR_vec Fy_y, TR_vec Fy_z, TR_vec Fz_x, TR_vec Fz_y, TR_vec Fz_z, TR_vec resx, TR_vec resy, TR_vec resz)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::negate3(TC_vec v_x, TC_vec v_y, TC_vec v_z)
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
void nonlinear_operators::Kolmogorov_3D_ker<TR, TR_vec, TC, TC_vec>::copy3(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z)
{

    copy3_kernel<TC, TC_vec><<<dimGrid1C, dimBlock1>>>(u_x, u_y, u_z, Nx*Ny*Mz, v_x, v_y, v_z);

}
#endif // __KOLMOGOROV_3D_IMPL_CUH__