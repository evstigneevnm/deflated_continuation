#ifndef __PLOT_SOLUTION_2D_H__
#define __PLOT_SOLUTION_2D_H__



#include <stdio.h>
#include <stdlib.h>

#include <common/macros.h>


namespace nonlinear_operators
{


template<class VecOpsR>
class plot_solution
{
private:
    typedef typename VecOpsR::scalar_type T;
    typedef typename VecOpsR::vector_type T_vec;

    size_t Nx, Ny;
    T Lx, Ly, dx, dy, dz;
    VecOpsR* vec_ops;
public:
    
    plot_solution(VecOpsR* vec_ops_, size_t Nx_, size_t Ny_, T Lx_, T Ly_):
    vec_ops(vec_ops_),
    Nx(Nx_), Ny(Ny_), Lx(Lx_), Ly(Ly_)
    {
        dx = Lx/T(Nx);
        dy = Ly/T(Ny);
        dz = std::min(dx,dy);
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
    



private:
    

    void write_out_file_pos(const std::string& f_name, const T_vec& U, int Nx, int Ny, int what = 2)
    {
        T Xmin=0.0, Ymin=0.0, Zmin=0.0;
        



        FILE *stream;


        stream=fopen(f_name.c_str(), "w" );


        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");

        int l = 0;
        for(int j=0;j<Nx;j++)
        for(int k=0;k<Ny;k++)
        {
            T par=0.0;
            T par_mm=0.0;
            T par_pm=0.0;
            T par_pp=0.0;
            T par_mp=0.0;

            par=U[I2P(j,k)];
            if(what==2)
            { 

                par_mm=0.25f*(U[I2P(j,k)]+U[I2P(j-1,k)]+U[I2P(j,k-1)]+U[I2P(j-1,k-1)]);
                par_pm=0.25f*(U[I2P(j,k)]+U[I2P(j+1,k)]+U[I2P(j,k-1)]+U[I2P(j+1,k-1)]);
                par_pp=0.25f*(U[I2P(j,k)]+U[I2P(j+1,k)]+U[I2P(j,k+1)]+U[I2P(j+1,k+1)]);
                par_mp=0.25f*(U[I2P(j,k)]+U[I2P(j-1,k)]+U[I2P(j,k+1)]+U[I2P(j-1,k+1)]);
            }
                    
                

                    fprintf( stream, "SH(%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)",
                    Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
                    Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
                    Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz,
                    Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz);

                    if(what==2){
                        fprintf( stream,"{");
                        fprintf(stream, "%e,",par_mm);
                        fprintf(stream, "%e,",par_pm);
                        fprintf(stream, "%e,",par_pp);
                        fprintf(stream, "%e,",par_mp);
                        fprintf(stream, "%e,",par_mm);
                        fprintf(stream, "%e,",par_pm);
                        fprintf(stream, "%e,",par_pp);
                        fprintf(stream, "%e",par_mp);
                        fprintf(stream, "};\n");
                    }
                    else if(what==1){
                        fprintf( stream,"{");
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e,",par);
                        fprintf(stream, "%e",par);
                        fprintf(stream, "};\n");
                    }

        }   

            

        fprintf( stream, "};");

        fclose(stream);


    }



    void write_out_file_pos(const std::string& f_name, const T_vec& ux, const T_vec&  uy,  int Nx, int Ny, int what = 2)
    {


        T Xmin=0.0, Ymin=0.0, Zmin=0.0;


        FILE *stream;


        stream=fopen(f_name.c_str(), "w" );


        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");

        int l = 0;
        for(int j=0;j<Nx;j++)
        for(int k=0;k<Ny;k++)
        {
            T par_x=0.0,par_y=0.0,par_z=0.0;
            T par_x_mm=0.0;
            T par_x_pm=0.0;
            T par_x_pp=0.0;
            T par_x_mp=0.0;
            T par_y_mm=0.0;
            T par_y_pm=0.0;
            T par_y_pp=0.0;
            T par_y_mp=0.0;
            T par_z_mmm=0.0;
            T par_z_pmm=0.0;
            T par_z_ppm=0.0;
            T par_z_ppp=0.0;
            T par_z_mpp=0.0;
            T par_z_mmp=0.0;
            T par_z_pmp=0.0;
            T par_z_mpm=0.0;

            
            par_x=ux[I2P(j,k)];
            par_y=uy[I2P(j,k)];
            par_z=0.0;
            par_x_mm=0.25f*(ux[I2P(j,k)]+ux[I2P(j-1,k)]+ux[I2P(j,k-1)]+ux[I2P(j-1,k-1)]);
            par_x_pm=0.25f*(ux[I2P(j,k)]+ux[I2P(j+1,k)]+ux[I2P(j,k-1)]+ux[I2P(j+1,k-1)]);
            par_x_pp=0.25f*(ux[I2P(j,k)]+ux[I2P(j+1,k)]+ux[I2P(j,k+1)]+ux[I2P(j+1,k+1)]);
            par_x_mp=0.25f*(ux[I2P(j,k)]+ux[I2P(j-1,k)]+ux[I2P(j,k+1)]+ux[I2P(j-1,k+1)]);
            
            par_y_mm=0.25f*(uy[I2P(j,k)]+uy[I2P(j-1,k)]+uy[I2P(j,k-1)]+uy[I2P(j-1,k-1)]);
            par_y_pm=0.25f*(uy[I2P(j,k)]+uy[I2P(j+1,k)]+uy[I2P(j,k-1)]+uy[I2P(j+1,k-1)]);
            par_y_pp=0.25f*(uy[I2P(j,k)]+uy[I2P(j+1,k)]+uy[I2P(j,k+1)]+uy[I2P(j+1,k+1)]);
            par_y_mp=0.25f*(uy[I2P(j,k)]+uy[I2P(j-1,k)]+uy[I2P(j,k+1)]+uy[I2P(j-1,k+1)]);
            
                    
  
            fprintf( stream, "VH(%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)",
                    Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l-0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz,
                    Xmin+dx*j+0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz,
                    Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l+0.5*dz, 
                    Xmin+dx*j-0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz, 
                    Xmin+dx*j+0.5*dx, Ymin+dy*k-0.5*dy, Zmin+dz*l+0.5*dz,
                    Xmin+dx*j-0.5*dx, Ymin+dy*k+0.5*dy, Zmin+dz*l-0.5*dz);

            if(what==2){
                fprintf( stream,"{");
                fprintf(stream, "%e,%e,%e,",par_x_mm,par_y_mm,par_z_mmm);
                fprintf(stream, "%e,%e,%e,",par_x_pm,par_y_pm,par_z_pmm);
                fprintf(stream, "%e,%e,%e,",par_x_pp,par_y_pp,par_z_ppm);
                fprintf(stream, "%e,%e,%e,",par_x_pp,par_y_pp,par_z_ppp);
                fprintf(stream, "%e,%e,%e,",par_x_mp,par_y_mp,par_z_mpp);
                fprintf(stream, "%e,%e,%e,",par_x_mm,par_y_mm,par_z_mmp);
                fprintf(stream, "%e,%e,%e,",par_x_pm,par_y_pm,par_z_pmp);
                fprintf(stream, "%e,%e,%e",par_x_mp,par_y_mp,par_z_mpm);
                fprintf(stream, "};\n");
            }
            else if(what==1){
                fprintf( stream,"{");
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e,",par_x,par_y,par_z);
                fprintf(stream, "%e,%e,%e",par_x,par_y,par_z);
                fprintf(stream, "};\n");
            }

        }   

        fprintf( stream, "};");

        fclose(stream);

    }


};

}


#endif // __PLOT_SOLUTION_2D_H__