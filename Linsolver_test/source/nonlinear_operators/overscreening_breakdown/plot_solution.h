#ifndef __PLOT_SOLUTION_2D_H__
#define __PLOT_SOLUTION_2D_H__



#include <stdio.h>
#include <stdlib.h>

#include <common/macros.h>


namespace nonlinear_operators
{


template<class VectorOperations>
class plot_solution
{
private:
    typedef typename VectorOperations::scalar_type T;
    typedef typename VectorOperations::vector_type T_vec;

    size_t Nx, Ny;
    T Lx, Ly, dx, dy, dz;
    VectorOperations* vec_ops_;
public:
    
    plot_solution(VectorOperations* vec_ops):
    vec_ops_(vec_ops),
    Nx(Nx_), Ny(Ny_), Lx(Lx_), Ly(Ly_)
    {

    }
    ~plot_solution()
    {
        
    }

    void write_to_disk(const std::string& f_name, const T_vec& u_d, int what = 2)
    {
        
        size_t sz = Nx*Ny;
        T_vec u_h = (T_vec) malloc(sizeof(T)*sz);

        device_2_host_cpy<T>(u_h, u_d, sz);


        write_out_file_pos(f_name, u_h, Nx, Ny, what);

        free(u_h);
    }

    void write_to_disk(const std::string& f_name, const T_vec& ux_d, const T_vec& uy_d,  int what = 2)
    {
        size_t sz = Nx*Ny;
        T_vec ux_h = (T_vec) malloc(sizeof(T)*sz);
        T_vec uy_h = (T_vec) malloc(sizeof(T)*sz);

        device_2_host_cpy<T>(ux_h, ux_d, sz);
        device_2_host_cpy<T>(uy_h, uy_d, sz);


        write_out_file_pos(f_name, ux_h, uy_h, Nx, Ny, what);

        free(ux_h);
        free(uy_h);
    }
    
    void write_to_disk_plain(const std::string& f_name, const T_vec& u_d)
    {
        
        size_t sz = Nx*Ny;
        T_vec u_h = (T_vec) malloc(sizeof(T)*sz);

        device_2_host_cpy<T>(u_h, u_d, sz);

        FILE *stream;
        stream=fopen(f_name.c_str(), "w" );
        for(int j=0;j<Nx;j++)
        {
            for(int k=0;k<Ny-1;k++)
            {
                fprintf(stream, "%lf ", (double)u_h[I2P(j,k)]);
            }
            fprintf(stream, "%lf\n", (double)u_h[I2P(j,Ny-1)]);
        }
        fclose(stream);
        free(u_h);
    }



};

}


#endif // __PLOT_SOLUTION_2D_H__