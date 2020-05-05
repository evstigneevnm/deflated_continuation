#ifndef __PLOT_SOLUTION_H__
#define __PLOT_SOLUTION_H__



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

    size_t Nx, Ny, Nz;
    T Lx, Ly, Lz, dx, dy, dz;
    VecOpsR* vec_ops;
public:
    
    plot_solution(VecOpsR* vec_ops_, size_t Nx_, size_t Ny_, size_t Nz_, T Lx_, T Ly_, T Lz_):
    vec_ops(vec_ops_),
    Nx(Nx_), Ny(Ny_), Nz(Nz_), Lx(Lx_), Ly(Ly_), Lz(Lz_)
    {
        dx = Lx/T(Nx);
        dy = Ly/T(Ny);
        dz = Lz/T(Nz);
    }
    ~plot_solution()
    {

    }

    void write_to_disk(const std::string& f_name, const T_vec& u_d, int what = 2)
    {
        
        size_t sz = Nx*Ny*Nz;
        T_vec u_h = (T_vec) malloc(sizeof(T)*sz);

        device_2_host_cpy<T>(u_h, u_d, sz);


        write_out_file_pos(f_name, u_h, Nx, Ny, Nz, what);

        free(u_h);
    }

    void write_to_disk(const std::string& f_name, const T_vec& ux_d, const T_vec& uy_d, const T_vec& uz_d, int what = 2)
    {
        size_t sz = Nx*Ny*Nz;
        T_vec ux_h = (T_vec) malloc(sizeof(T)*sz);
        T_vec uy_h = (T_vec) malloc(sizeof(T)*sz);
        T_vec uz_h = (T_vec) malloc(sizeof(T)*sz);

        device_2_host_cpy<T>(ux_h, ux_d, sz);
        device_2_host_cpy<T>(uy_h, uy_d, sz);
        device_2_host_cpy<T>(uz_h, uz_d, sz);


        write_out_file_pos(f_name, ux_h, uy_h, uz_h, Nx, Ny, Nz, what);

        free(ux_h);
        free(uy_h);
        free(uz_h);
    }
    


private:
    

    void write_out_file_pos(const std::string& f_name, const T_vec& U, int Nx, int Ny, int Nz, int what = 2)
    {
        T Xmin=0.0, Ymin=0.0, Zmin=0.0;




        FILE *stream;


        stream=fopen(f_name.c_str(), "w" );


        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");


        for(int j=0;j<Nx;j++)
        for(int k=0;k<Ny;k++)
        for(int l=0;l<Nz;l++)
        {
            T par=0.0;
            T par_mmm=0.0;
            T par_pmm=0.0;
            T par_ppm=0.0;
            T par_ppp=0.0;
            T par_mpp=0.0;
            T par_mmp=0.0;
            T par_pmp=0.0;
            T par_mpm=0.0;

            par=U[I3P(j,k,l)];
            if(what==2){ 

                par_mmm=0.125f*(U[I3P(j,k,l)]+U[I3P(j-1,k,l)]+U[I3P(j,k-1,l)]+U[I3P(j,k,l-1)]+U[I3P(j-1,k-1,l)]+U[I3P(j,k-1,l-1)]+U[I3P(j-1,k,l-1)]+U[I3P(j-1,k-1,l-1)]);
                par_pmm=0.125f*(U[I3P(j,k,l)]+U[I3P(j+1,k,l)]+U[I3P(j,k-1,l)]+U[I3P(j,k,l-1)]+U[I3P(j+1,k-1,l)]+U[I3P(j,k-1,l-1)]+U[I3P(j+1,k,l-1)]+U[I3P(j+1,k-1,l-1)]);
                par_ppm=0.125f*(U[I3P(j,k,l)]+U[I3P(j+1,k,l)]+U[I3P(j,k+1,l)]+U[I3P(j,k,l-1)]+U[I3P(j+1,k+1,l)]+U[I3P(j,k+1,l-1)]+U[I3P(j+1,k,l-1)]+U[I3P(j+1,k+1,l-1)]);
                par_ppp=0.125f*(U[I3P(j,k,l)]+U[I3P(j+1,k,l)]+U[I3P(j,k+1,l)]+U[I3P(j,k,l+1)]+U[I3P(j+1,k+1,l)]+U[I3P(j,k+1,l+1)]+U[I3P(j+1,k,l+1)]+U[I3P(j+1,k+1,l+1)]);
                par_mpp=0.125f*(U[I3P(j,k,l)]+U[I3P(j-1,k,l)]+U[I3P(j,k+1,l)]+U[I3P(j,k,l+1)]+U[I3P(j-1,k+1,l)]+U[I3P(j,k+1,l+1)]+U[I3P(j-1,k,l+1)]+U[I3P(j-1,k+1,l+1)]);
                par_mmp=0.125f*(U[I3P(j,k,l)]+U[I3P(j-1,k,l)]+U[I3P(j,k-1,l)]+U[I3P(j,k,l+1)]+U[I3P(j-1,k-1,l)]+U[I3P(j,k-1,l+1)]+U[I3P(j-1,k,l+1)]+U[I3P(j-1,k-1,l+1)]);
                par_pmp=0.125f*(U[I3P(j,k,l)]+U[I3P(j+1,k,l)]+U[I3P(j,k-1,l)]+U[I3P(j,k,l+1)]+U[I3P(j+1,k-1,l)]+U[I3P(j,k-1,l+1)]+U[I3P(j+1,k,l+1)]+U[I3P(j+1,k-1,l+1)]);
                par_mpm=0.125f*(U[I3P(j,k,l)]+U[I3P(j-1,k,l)]+U[I3P(j,k+1,l)]+U[I3P(j,k,l-1)]+U[I3P(j-1,k+1,l)]+U[I3P(j,k+1,l-1)]+U[I3P(j-1,k,l-1)]+U[I3P(j-1,k+1,l-1)]);
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
                        fprintf(stream, "%e,",par_mmm);
                        fprintf(stream, "%e,",par_pmm);
                        fprintf(stream, "%e,",par_ppm);
                        fprintf(stream, "%e,",par_mpm);
                        fprintf(stream, "%e,",par_mmp);
                        fprintf(stream, "%e,",par_pmp);
                        fprintf(stream, "%e,",par_ppp);
                        fprintf(stream, "%e",par_mpp);
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



    void write_out_file_pos(const std::string& f_name, const T_vec& ux, const T_vec&  uy, const T_vec&  uz, int Nx, int Ny, int Nz, int what = 2)
    {


        T Xmin=0.0, Ymin=0.0, Zmin=0.0;

        FILE *stream;


        stream=fopen(f_name.c_str(), "w" );


        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");


        for(int j=0;j<Nx;j++)
        for(int k=0;k<Ny;k++)
        for(int l=0;l<Nz;l++){
            T par_x=0.0,par_y=0.0,par_z=0.0;
            T par_x_mmm=0.0;
            T par_x_pmm=0.0;
            T par_x_ppm=0.0;
            T par_x_ppp=0.0;
            T par_x_mpp=0.0;
            T par_x_mmp=0.0;
            T par_x_pmp=0.0;
            T par_x_mpm=0.0;
            T par_y_mmm=0.0;
            T par_y_pmm=0.0;
            T par_y_ppm=0.0;
            T par_y_ppp=0.0;
            T par_y_mpp=0.0;
            T par_y_mmp=0.0;
            T par_y_pmp=0.0;
            T par_y_mpm=0.0;
            T par_z_mmm=0.0;
            T par_z_pmm=0.0;
            T par_z_ppm=0.0;
            T par_z_ppp=0.0;
            T par_z_mpp=0.0;
            T par_z_mmp=0.0;
            T par_z_pmp=0.0;
            T par_z_mpm=0.0;

            
            par_x=ux[I3P(j,k,l)];
            par_y=uy[I3P(j,k,l)];
            par_z=uz[I3P(j,k,l)];
            par_x_mmm=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j-1,k,l)]+ux[I3P(j,k-1,l)]+ux[I3P(j,k,l-1)]+ux[I3P(j-1,k-1,l)]+ux[I3P(j,k-1,l-1)]+ux[I3P(j-1,k,l-1)]+ux[I3P(j-1,k-1,l-1)]);
            par_x_pmm=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j+1,k,l)]+ux[I3P(j,k-1,l)]+ux[I3P(j,k,l-1)]+ux[I3P(j+1,k-1,l)]+ux[I3P(j,k-1,l-1)]+ux[I3P(j+1,k,l-1)]+ux[I3P(j+1,k-1,l-1)]);
            par_x_ppm=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j+1,k,l)]+ux[I3P(j,k+1,l)]+ux[I3P(j,k,l-1)]+ux[I3P(j+1,k+1,l)]+ux[I3P(j,k+1,l-1)]+ux[I3P(j+1,k,l-1)]+ux[I3P(j+1,k+1,l-1)]);
            par_x_ppp=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j+1,k,l)]+ux[I3P(j,k+1,l)]+ux[I3P(j,k,l+1)]+ux[I3P(j+1,k+1,l)]+ux[I3P(j,k+1,l+1)]+ux[I3P(j+1,k,l+1)]+ux[I3P(j+1,k+1,l+1)]);
            par_x_mpp=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j-1,k,l)]+ux[I3P(j,k+1,l)]+ux[I3P(j,k,l+1)]+ux[I3P(j-1,k+1,l)]+ux[I3P(j,k+1,l+1)]+ux[I3P(j-1,k,l+1)]+ux[I3P(j-1,k+1,l+1)]);
            par_x_mmp=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j-1,k,l)]+ux[I3P(j,k-1,l)]+ux[I3P(j,k,l+1)]+ux[I3P(j-1,k-1,l)]+ux[I3P(j,k-1,l+1)]+ux[I3P(j-1,k,l+1)]+ux[I3P(j-1,k-1,l+1)]);
            par_x_pmp=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j+1,k,l)]+ux[I3P(j,k-1,l)]+ux[I3P(j,k,l+1)]+ux[I3P(j+1,k-1,l)]+ux[I3P(j,k-1,l+1)]+ux[I3P(j+1,k,l+1)]+ux[I3P(j+1,k-1,l+1)]);
            par_x_mpm=0.125f*(ux[I3P(j,k,l)]+ux[I3P(j-1,k,l)]+ux[I3P(j,k+1,l)]+ux[I3P(j,k,l-1)]+ux[I3P(j-1,k+1,l)]+ux[I3P(j,k+1,l-1)]+ux[I3P(j-1,k,l-1)]+ux[I3P(j-1,k+1,l-1)]);
            
            par_y_mmm=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j-1,k,l)]+uy[I3P(j,k-1,l)]+uy[I3P(j,k,l-1)]+uy[I3P(j-1,k-1,l)]+uy[I3P(j,k-1,l-1)]+uy[I3P(j-1,k,l-1)]+uy[I3P(j-1,k-1,l-1)]);
            par_y_pmm=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j+1,k,l)]+uy[I3P(j,k-1,l)]+uy[I3P(j,k,l-1)]+uy[I3P(j+1,k-1,l)]+uy[I3P(j,k-1,l-1)]+uy[I3P(j+1,k,l-1)]+uy[I3P(j+1,k-1,l-1)]);
            par_y_ppm=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j+1,k,l)]+uy[I3P(j,k+1,l)]+uy[I3P(j,k,l-1)]+uy[I3P(j+1,k+1,l)]+uy[I3P(j,k+1,l-1)]+uy[I3P(j+1,k,l-1)]+uy[I3P(j+1,k+1,l-1)]);
            par_y_ppp=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j+1,k,l)]+uy[I3P(j,k+1,l)]+uy[I3P(j,k,l+1)]+uy[I3P(j+1,k+1,l)]+uy[I3P(j,k+1,l+1)]+uy[I3P(j+1,k,l+1)]+uy[I3P(j+1,k+1,l+1)]);
            par_y_mpp=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j-1,k,l)]+uy[I3P(j,k+1,l)]+uy[I3P(j,k,l+1)]+uy[I3P(j-1,k+1,l)]+uy[I3P(j,k+1,l+1)]+uy[I3P(j-1,k,l+1)]+uy[I3P(j-1,k+1,l+1)]);
            par_y_mmp=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j-1,k,l)]+uy[I3P(j,k-1,l)]+uy[I3P(j,k,l+1)]+uy[I3P(j-1,k-1,l)]+uy[I3P(j,k-1,l+1)]+uy[I3P(j-1,k,l+1)]+uy[I3P(j-1,k-1,l+1)]);
            par_y_pmp=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j+1,k,l)]+uy[I3P(j,k-1,l)]+uy[I3P(j,k,l+1)]+uy[I3P(j+1,k-1,l)]+uy[I3P(j,k-1,l+1)]+uy[I3P(j+1,k,l+1)]+uy[I3P(j+1,k-1,l+1)]);
            par_y_mpm=0.125f*(uy[I3P(j,k,l)]+uy[I3P(j-1,k,l)]+uy[I3P(j,k+1,l)]+uy[I3P(j,k,l-1)]+uy[I3P(j-1,k+1,l)]+uy[I3P(j,k+1,l-1)]+uy[I3P(j-1,k,l-1)]+uy[I3P(j-1,k+1,l-1)]);
            
                    
            par_z_mmm=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j-1,k,l)]+uz[I3P(j,k-1,l)]+uz[I3P(j,k,l-1)]+uz[I3P(j-1,k-1,l)]+uz[I3P(j,k-1,l-1)]+uz[I3P(j-1,k,l-1)]+uz[I3P(j-1,k-1,l-1)]);
            par_z_pmm=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j+1,k,l)]+uz[I3P(j,k-1,l)]+uz[I3P(j,k,l-1)]+uz[I3P(j+1,k-1,l)]+uz[I3P(j,k-1,l-1)]+uz[I3P(j+1,k,l-1)]+uz[I3P(j+1,k-1,l-1)]);
            par_z_ppm=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j+1,k,l)]+uz[I3P(j,k+1,l)]+uz[I3P(j,k,l-1)]+uz[I3P(j+1,k+1,l)]+uz[I3P(j,k+1,l-1)]+uz[I3P(j+1,k,l-1)]+uz[I3P(j+1,k+1,l-1)]);
            par_z_ppp=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j+1,k,l)]+uz[I3P(j,k+1,l)]+uz[I3P(j,k,l+1)]+uz[I3P(j+1,k+1,l)]+uz[I3P(j,k+1,l+1)]+uz[I3P(j+1,k,l+1)]+uz[I3P(j+1,k+1,l+1)]);
            par_z_mpp=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j-1,k,l)]+uz[I3P(j,k+1,l)]+uz[I3P(j,k,l+1)]+uz[I3P(j-1,k+1,l)]+uz[I3P(j,k+1,l+1)]+uz[I3P(j-1,k,l+1)]+uz[I3P(j-1,k+1,l+1)]);
            par_z_mmp=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j-1,k,l)]+uz[I3P(j,k-1,l)]+uz[I3P(j,k,l+1)]+uz[I3P(j-1,k-1,l)]+uz[I3P(j,k-1,l+1)]+uz[I3P(j-1,k,l+1)]+uz[I3P(j-1,k-1,l+1)]);
            par_z_pmp=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j+1,k,l)]+uz[I3P(j,k-1,l)]+uz[I3P(j,k,l+1)]+uz[I3P(j+1,k-1,l)]+uz[I3P(j,k-1,l+1)]+uz[I3P(j+1,k,l+1)]+uz[I3P(j+1,k-1,l+1)]);
            par_z_mpm=0.125f*(uz[I3P(j,k,l)]+uz[I3P(j-1,k,l)]+uz[I3P(j,k+1,l)]+uz[I3P(j,k,l-1)]+uz[I3P(j-1,k+1,l)]+uz[I3P(j,k+1,l-1)]+uz[I3P(j-1,k,l-1)]+uz[I3P(j-1,k+1,l-1)]);

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
                fprintf(stream, "%e,%e,%e,",par_x_mmm,par_y_mmm,par_z_mmm);
                fprintf(stream, "%e,%e,%e,",par_x_pmm,par_y_pmm,par_z_pmm);
                fprintf(stream, "%e,%e,%e,",par_x_ppm,par_y_ppm,par_z_ppm);
                fprintf(stream, "%e,%e,%e,",par_x_ppp,par_y_ppp,par_z_ppp);
                fprintf(stream, "%e,%e,%e,",par_x_mpp,par_y_mpp,par_z_mpp);
                fprintf(stream, "%e,%e,%e,",par_x_mmp,par_y_mmp,par_z_mmp);
                fprintf(stream, "%e,%e,%e,",par_x_pmp,par_y_pmp,par_z_pmp);
                fprintf(stream, "%e,%e,%e",par_x_mpm,par_y_mpm,par_z_mpm);
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


#endif // __PLOT_SOLUTION_H__