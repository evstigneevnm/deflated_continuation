#ifndef __BIFURCATION_DIAGRAM_H__
#define __BIFURCATION_DIAGRAM_H__

/**
*
*   Main container that holds all bifurcaiton diagram curves.
*   
*   The curves are typed by the Curve template parameter.
*   Can be used with arbitrary curves, but inlined curves are more preferable.
*   Helper class is also included to avoid memory usage on the curves during interpolation
*/

#include <vector>


namespace container
{

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class Newton, class SolutionStorage,  class Curve, class CurveHelper>
class bifurcation_diagram
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

private:
    typedef CurveHelper cont_help_t;

    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperator* nonlin_op;
    Newton* newton;
    cont_help_t* cont_help;
    unsigned int skip_output;
    std::string directory;

public:
    bifurcation_diagram(VectorOperations*& vec_ops_, VectorFileOperations*& vec_files_, Log*& log_, NonlinearOperator*& nlin_op_, Newton*& newton_, const std::string& directory_ = "dat_files", unsigned int skip_output_ = 10):
    vec_ops(vec_ops_),
    file_ops(vec_files_),
    log(log_),
    nonlin_op(nlin_op_),
    newton(newton_),
    skip_output(skip_output_),
    directory(directory_)
    {
        cont_help = new cont_help_t(vec_ops);
        curve_number = -1;

    }

    ~bifurcation_diagram()
    {
        delete cont_help;
    }

    void get_current_ref(Curve*& curve_ref)
    {
        std::cout << "reference to the curve inside = " << &curve_container.back() << std::endl;
        curve_ref = &curve_container.back();
    }

    void init_new_curve()
    {
        curve_number++;
        curve_container.emplace_back( Curve(vec_ops, file_ops, log, nonlin_op, newton, curve_number, directory, cont_help, skip_output) );
    }

    void find_intersection(const T& lambda_star, SolutionStorage*& solution_vector)
    {
        for(auto &x: curve_container)
        {
            try
            {
                x.find_intersection(lambda_star, solution_vector);
            }
            catch(const std::exception& e)
            {
                log->info_f("bifurcation_diagram::find_intersection: %s\n", e.what());
            }
        }

    }


private:
    std::vector<Curve> curve_container;
    unsigned int curve_number = -1;
    
};





}

#endif // __BIFURCATION_DIAGRAM_H__