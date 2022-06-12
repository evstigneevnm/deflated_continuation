#ifndef __ABC_FLOW_KER_H__
#define __ABC_FLOW_KER_H__

/**
*   Axillary class of funcitons for Kolmogorov 3D flow used in the Kolmogorov_3D class.
*   The class implements explicit template specialization.
*/
#include <cstdio> //for printf
#include <cuda_runtime.h>
#include <cmath>

namespace nonlinear_operators
{

template<typename TR, typename TR_vec, typename TC, typename TC_vec>
struct abc_flow_ker
{
    abc_flow_ker(size_t Nx_, size_t Ny_, size_t Nz_, size_t Mz_, unsigned int BLOCKSIZE_x_, unsigned int BLOCKSIZE_y_):
    Nx(Nx_),Ny(Ny_),Nz(Nz_),Mz(Mz_), BLOCKSIZE_x(BLOCKSIZE_x_), BLOCKSIZE_y(BLOCKSIZE_y_)
    {
        BLOCKSIZE = BLOCKSIZE_x*BLOCKSIZE_y;
        NR = Nx*Ny*Nz;
        NC = Nx*Ny*Mz;
        N = 6*(NC - 1);
        calculate_cuda_grid();
    }

    ~abc_flow_ker()
    {

    }

    void Laplace_Fourier(TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec Laplace);

    void force_Fourier_sin_cos(int n_y, int n_z, TR scale_const, TC_vec force_x, TC_vec force_y, TC_vec force_z);    

    void force_ABC(TR_vec force_x, TR_vec force_y, TR_vec force_z);

    void vec2complex(TR_vec v_in, TC_vec u_x, TC_vec u_y, TC_vec u_z);

    void complex2vec(TC_vec u_x, TC_vec u_y, TC_vec u_z, TR_vec v_out);

    void apply_grad(TC_vec v_in,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void apply_div(TC_vec u_x, TC_vec u_y, TC_vec u_z,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_out);
    
    void apply_Laplace(TR coeff, TC_vec Laplace, TC_vec ux, TC_vec uy, TC_vec uz);

    void apply_iLaplace(TC_vec Laplace, TC_vec v, TR coeff = TR(1.0));

    void apply_iLaplace3(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff = TR(1.0));

    void apply_projection(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec Laplace,  TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void apply_smooth(TR tau, TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void imag_vector(TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void apply_abs(TR_vec ux, TR_vec uy, TR_vec uz, TR_vec v);
    void apply_abs(size_t Nx, size_t Ny, size_t Nz, TR_vec ux, TR_vec uy, TR_vec uz, TR_vec v);
    void apply_scale_inplace(size_t Nx_l, size_t Ny_l, size_t Nz_l, TR scale, TR_vec ux, TR_vec uy, TR_vec uz);
    void add_mul3(TR alpha, TR_vec u_x, TR_vec u_y, TR_vec u_z, TR_vec v_x, TR_vec v_y, TR_vec v_z);

    void add_mul3(TC alpha, TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void apply_mask(TC_vec mask_2_3);

    void apply_grad3(TC_vec u1, TC_vec u2, TC_vec u3, TC_vec mask, TC_vec grad_x, TC_vec grad_y, TC_vec grad_z, TC_vec v1_x, TC_vec v1_y, TC_vec v1_z, TC_vec v2_x, TC_vec v2_y, TC_vec v2_z, TC_vec v3_x, TC_vec v3_y, TC_vec v3_z);

    void multiply_advection(TR_vec Vx, TR_vec Vy, TR_vec Vz, TR_vec Fx_x, TR_vec Fx_y, TR_vec Fx_z, TR_vec Fy_x, TR_vec Fy_y, TR_vec Fy_z, TR_vec Fz_x, TR_vec Fz_y, TR_vec Fz_z, TR_vec resx, TR_vec resy, TR_vec resz);

    void negate3(TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void copy3(TC_vec u_x, TC_vec u_y, TC_vec u_z, TC_vec v_x, TC_vec v_y, TC_vec v_z);

    void B_ABC_exact(TR coeff, TR_vec ux, TR_vec uy, TR_vec uz);

    void apply_iLaplace3_plus_E(TC_vec Laplace, TC_vec v_x, TC_vec v_y, TC_vec v_z, TR coeff, TR a, TR b);
    
    void convert_size(size_t Nx_dest, size_t Ny_dest, size_t Mz_dest, TR scale, TC_vec ux_src_hat, TC_vec uy_src_hat, TC_vec uz_src_hat, TC_vec ux_dest_hat, TC_vec uy_dest_hat, TC_vec uz_dest_hat);

private:
    unsigned int BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE;
    size_t Nx, Ny, Nz, Mz;
    size_t N, NR, NC;

    //for 1D-like operations:
    dim3 dimBlock1;
    dim3 dimGrid1;
    dim3 dimGrid1R;
    dim3 dimGrid1C;
    //for 3D operations:
    dim3 dimBlockN;
    dim3 dimGridNR;
    dim3 dimGridNC;



    void calculate_cuda_grid()
    {
        
        printf("abc flow template specialization class:\n");
        //1D
        dim3 s_dimBlock1(BLOCKSIZE);
        dimBlock1 = s_dimBlock1;

        int blocks_x;
        //1D blocks 
        blocks_x=(N+BLOCKSIZE)/BLOCKSIZE;
        dim3 s_dimGrid1(blocks_x);
        dimGrid1=s_dimGrid1;
        
        blocks_x=(NR+BLOCKSIZE)/BLOCKSIZE;
        dim3 s_dimGrid1R(blocks_x);
        dimGrid1R=s_dimGrid1R;

        blocks_x=(NC+BLOCKSIZE)/BLOCKSIZE;
        dim3 s_dimGrid1C(blocks_x);
        dimGrid1C=s_dimGrid1C;

        //blocks for 3D vectors
        dim3 s_dimBlockN(BLOCKSIZE_x, BLOCKSIZE_y, 1);
        dimBlockN = s_dimBlockN;

        // grid for 3D Real vectors
        unsigned int k1, k2;
        unsigned int nthreads = BLOCKSIZE_x * BLOCKSIZE_y;
        unsigned int nblocks = ( NR + nthreads -1 )/nthreads ;
        double db_nblocks = double(nblocks);
        k1 = (unsigned int) double(std::sqrt(db_nblocks) ) ;
        k2 = (unsigned int) std::ceil( db_nblocks/( double(k1) )) ;

        dim3 s_dimGridNR( k2, k1, 1 );
        dimGridNR = s_dimGridNR;
        // debug:
        printf("    Reals grids: dimGrid: %dX%dX%d, dimBlock: %dX%dX%d. \n", dimGridNR.x, dimGridNR.y, dimGridNR.z, dimBlockN.x, dimBlockN.y, dimBlockN.z);

        //grid for 3D Complex vectors
        nblocks = ( NC + nthreads -1 )/nthreads ;
        db_nblocks = double(nblocks);
        k1 = (unsigned int) double(std::sqrt(db_nblocks) ) ;
        k2 = (unsigned int) std::ceil( db_nblocks/( double(k1) )) ;  

        dim3 s_dimGridNC( k2, k1, 1 );  
        dimGridNC = s_dimGridNC;
        // debug:        
        printf("    Complex grids: dimGrid: %dX%dX%d, dimBlock: %dX%dX%d. \n", dimGridNC.x, dimGridNC.y, dimGridNC.z, dimBlockN.x, dimBlockN.y, dimBlockN.z);
    }



};

}

#endif 