#ifndef __POS_BIF_DIAG_OUTPUT_H__
#define __POS_BIF_DIAG_OUTPUT_H__

/*
#include "taylor_model.h"

void    print_res_line(FILE *stream, double u1, double v1, double u2, double v2)
{
        fprintf( stream, "SL(%0.15e, %0.15e, %0.15e, %0.15e, %0.15e, %0.15e)",  u1, v1, 0.f,
                                                                                u2, v2, 0.f);
        fprintf( stream,"{");
        fprintf(stream, "%f,",1.f);
        fprintf(stream, "%f",1.f);
        fprintf(stream, "};\n");
}

void    print_res_curve(FILE *stream, double *u, double *v, int steps_n = 1)
{
        for (int i = 0;i < steps_n;++i) {
                double  u1_ = u[i], 
                        v1_ = v[i], 
                        u2_ = u[i+1],
                        v2_ = v[i+1];
                fprintf( stream, "SL(%0.15e, %0.15e, %0.15e, %0.15e, %0.15e, %0.15e)",  u1_, v1_, 0.f,
                                                                                        u2_, v2_, 0.f);
                fprintf( stream,"{");
                fprintf(stream, "%f,",1.f);
                fprintf(stream, "%f",1.f);
                fprintf(stream, "};\n");
        }
}

void    print_res_rect(FILE *stream, double u1, double v1, double u2, double v2)
{
        print_res_line(stream, u1, v1, u2, v1);
        print_res_line(stream, u2, v1, u2, v2);
        print_res_line(stream, u2, v2, u1, v2);
        print_res_line(stream, u1, v2, u1, v1);
}

void    print_res_connect_rects(FILE *stream, double u11, double v11, double u12, double v12, double u21, double v21, double u22, double v22)
{
        print_res_line(stream, u11, v11, u21, v21);
        print_res_line(stream, u11, v12, u21, v22);
        print_res_line(stream, u12, v11, u22, v21);
        print_res_line(stream, u12, v12, u22, v22);
}

void    print_res_connect_rects(FILE *stream, double *u1, double *v1, double *u2, double *v2, int steps_n)
{
        print_res_curve(stream, u1, v1, steps_n);
        print_res_curve(stream, u1, v2, steps_n);
        print_res_curve(stream, u2, v1, steps_n);
        print_res_curve(stream, u2, v2, steps_n);
}

/// print_corners == true prints normal TM with remainders and projections
/// print_corners == false prints just polynomial part with one line border 
/// and has no real meaning for tms with dimension larger then 2
/// default values for var1_i and var2_i were 1 and 2 respec.
template<int Deg,class Int,class Mem>
void    print_res(FILE *stream, const prooving_algos::taylor_model_operations<Deg,Int,Mem> &ops, 
                  const prooving_algos::taylor_model<Deg,Int,Mem> &x, int view_num, bool print_corners,
                  int var1_i, int var2_i)
{
        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, "%d", view_num);
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");

        prooving_algos::taylor_model<Deg,Int,Mem> tmp1, tmp2;
        ops.init_taylor_model(tmp1); ops.init_taylor_model(tmp2);
        ops.start_use_taylor_model(tmp1); ops.start_use_taylor_model(tmp2);
        
        int steps_n = 20;

        double  *u1 = new double [steps_n+1], 
                *v1 = new double [steps_n+1], 
                *u2 = new double [steps_n+1], 
                *v2 = new double [steps_n+1];

        for (int l = 0;l < 4;++l) {
                double u_begin, v_begin, u_end, v_end;
                switch (l) {
                        case 0:
                                u_begin =-1.; v_begin =-1.; u_end = 1.; v_end =-1.;
                        break;
                        case 1:
                                u_begin = 1.; v_begin =-1.; u_end = 1.; v_end = 1.;
                        break;
                        case 2:
                                u_begin = 1.; v_begin = 1.; u_end =-1.; v_end = 1.;
                        break;
                        case 3:
                                u_begin =-1.; v_begin = 1.; u_end =-1.; v_end =-1.;
                        break;
                }
                for (int s = 0;s < steps_n+1;++s) {
                        double  s1 = u_begin + s*(u_end-u_begin)/steps_n, 
                                s2 = v_begin + s*(v_end-v_begin)/steps_n;
                        ops.substitute(x,    var1_i, s1, tmp1);
                        ops.substitute(tmp1, var2_i, s2, tmp2);
                        if (print_corners)
                            ops.calc_external_bnds(tmp2);
                        else 
                            ops.calc_bnds(tmp2);

                        auto bnds_view = tmp2.bnds().create_view(true);

                        if (print_corners && ((s == 0)||(s == steps_n)))
                                print_res_rect(stream, bnds_view(0).lower(), bnds_view(1).lower(), 
                                                       bnds_view(0).upper(), bnds_view(1).upper());

                        if (print_corners) {
                            u1[s] = bnds_view(0).lower(); 
                            v1[s] = bnds_view(1).lower();
                            u2[s] = bnds_view(0).upper();
                            v2[s] = bnds_view(1).upper();
                        } else {
                            u1[s] = u2[s] = 0.5*(bnds_view(0).lower() + bnds_view(0).upper());
                            v1[s] = v2[s] = 0.5*(bnds_view(1).lower() + bnds_view(1).upper());
                        }
                }
                if (print_corners) 
                    print_res_connect_rects(stream, u1, v1, u2, v2, steps_n);
                else
                    print_res_curve(stream, u1, v1, steps_n);
        }

        delete []u1;
        delete []v1;
        delete []u2;
        delete []v2;

        fprintf( stream, "};\n");

        ops.stop_use_taylor_model(tmp1); ops.stop_use_taylor_model(tmp2);
        ops.free_taylor_model(tmp1); ops.free_taylor_model(tmp2);
}

*/
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

namespace file_operations
{

    template<class T, class DiagramPointType>
    void plot_diagram_no_stability(const std::string path_to_file_, const std::string file_name_, int curve_number_, const std::vector<DiagramPointType>& diag_, int coord1_ = 0, int coord2_ = 1, unsigned int prec=6)
    {
        std::string f_name = path_to_file_ + file_name_;
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("file_operations::plot_diagram_no_stability: error while opening file" + f_name);
        f << "View '" << file_name_ << "'" << std::endl << "{" << std::endl;
        T lambda_text = T(0.0);
        T coord0_text = T(0.0);
        T coord1_text = T(0.0);
        
        int num_points = diag_.size();
        
        T coord1_0 = diag_.at(0).vector_norms.at(coord1_);
        T coord2_0 = diag_.at(0).vector_norms.at(coord2_);

        for (auto &x: diag_)
        {
            T lambda = x.lambda;
            T coord1 = x.vector_norms.at(coord1_);
            T coord2 = x.vector_norms.at(coord2_);
            
            f << "SP(";
            f << std::setprecision(prec) << lambda << "," << coord1 << "," << coord2;
            f << "){" << curve_number_ << "};" << std::endl;
            lambda_text += lambda/T(num_points);
            coord0_text += coord1/T(num_points);
            coord1_text += coord2/T(num_points);
            
        }
        //T3(0, 0.11, 0, TextAttributes("Align", "Center", "Font", "Helvetica")){ "Hole" };
        //f << "T3(" << lambda_text << "," << coord0_text+coord1_0 << "," << coord1_text+coord2_0 << ", TextAttributes(\"Align\", \"Center\", \"Font\", \"Helvetica\")){\"" << std::to_string(curve_number_) << "\"};" << std::endl;

        f << "};";


        f.close();

    }


    template<class T, class DiagramPointType>
    void plot_diagram_stability(const std::string path_to_file_, const std::string file_name_, int curve_number_, const std::vector<DiagramPointType>& diag_, const std::vector<DiagramPointType>& diag_bif_, int coord1_ = 0, int coord2_ = 1, unsigned int prec=6)
    {
        std::string f_name = path_to_file_ + file_name_;
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("file_operations::plot_diagram_stability: error while opening file" + f_name);
        f << "View '" << file_name_ << "'" << std::endl << "{" << std::endl;
        T lambda_text = T(0.0);
        T coord0_text = T(0.0);
        T coord1_text = T(0.0);
        
        int num_points = diag_.size();
        
        T coord1_0 = diag_.at(0).vector_norms.at(coord1_);
        T coord2_0 = diag_.at(0).vector_norms.at(coord2_);

        for (auto &x: diag_)
        {
            T lambda = x.lambda;
            T coord1 = x.vector_norms.at(coord1_);
            T coord2 = x.vector_norms.at(coord2_);
            
            f << "SP(";
            f << std::setprecision(prec) << lambda << "," << coord1 << "," << coord2;
            f << "){" << x.dim_unstable << "};" << std::endl;
            lambda_text += lambda/T(num_points);
            coord0_text += coord1/T(num_points);
            coord1_text += coord2/T(num_points);
            
        }

        //add bifurcation points
        for (auto &x: diag_bif_)
        {
            T lambda = x.lambda;
            T coord1 = x.vector_norms.at(coord1_);
            T coord2 = x.vector_norms.at(coord2_);
            
            f << "SP(";
            f << std::setprecision(prec) << lambda << "," << coord1 << "," << coord2;
            f << "){" << x.dim_unstable << "};" << std::endl;

            //T3(0, 0.11, 0, TextAttributes("Align", "Center", "Font", "Helvetica")){ "Hole" };
            f << "T3(" << lambda << "," << coord1 << "," << coord2 << ", TextAttributes(\"Align\", \"Center\", \"Font\", \"Helvetica\")){\"" << std::to_string(x.dim_unstable) << "\"};" << std::endl;

        }


        f << "};";


        f.close();

    }

    template<class T, class DiagramPointType>
    void plot_diagram_stability_gnuplot(const std::string path_to_file_, const std::string file_name_, int curve_number_, const std::vector<DiagramPointType>& diag_, const std::vector<DiagramPointType>& diag_bif_, unsigned int prec=6)
    {
        std::string f_name = path_to_file_ + std::string("regular") + file_name_;
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("file_operations::plot_diagram_stability_gnuplot: error while opening file" + f_name);

        
        for (auto &x: diag_)
        {
            
            f << std::setprecision(prec) << x.lambda << " ";
            for(auto &y: x.vector_norms)
                f << y << " ";
            f << x.dim_unstable << " ";
            f << x.id_file_name; // needed to read image for bifurcation diagram
            f  << std::endl;

        }
        f.close();

        //add bifurcation points in a separate file
        f_name = path_to_file_ + std::string("singular") + file_name_;
        std::ofstream f_b(f_name.c_str(), std::ofstream::out);
        if (!f_b) throw std::runtime_error("file_operations::plot_diagram_stability_gnuplot: error while opening file" + f_name);

        
        for (auto &x: diag_bif_)
        {

            f_b << std::setprecision(prec) << x.lambda << " ";
            for(auto &y: x.vector_norms)
                f_b << y << " ";
            f_b << x.dim_unstable << " ";
            f_b << "B" << " "; //parse bifurcation type
            f_b << x.id_file_name; // needed to read image for bifurcation diagram
            f_b  << std::endl;

        }
        f_b.close();





    }




}

#endif //__POS_BIF_DIAG_OUTPUT_H__
