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
#include <string>
//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

namespace container
{

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class Newton, class SolutionStorage,  class Curve, class CurveHelper>
class bifurcation_diagram
{
private:
    friend class boost::serialization::access;

    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

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
    bifurcation_diagram(VectorOperations* vec_ops_, VectorFileOperations* vec_files_, Log* log_, NonlinearOperator* nlin_op_, Newton* newton_, const std::string& directory_ = {}, unsigned int skip_output_ = 10):
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

    bifurcation_diagram()
    {
        //void default constructor for boost serialization
    }

    ~bifurcation_diagram()
    {
        delete cont_help;
    }

    void get_current_ref(Curve*& curve_ref)
    {
        //std::cout << "reference to the curve inside = " << &curve_container.back() << std::endl;
        curve_container.back().set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
        curve_ref = &curve_container.back();
    }

    int current_curve()
    {
        return curve_container.size();
    }
    
    void init_new_curve()
    {
        curve_number++;
        curve_container.emplace_back( vec_ops, file_ops, log, nonlin_op, newton, curve_number, directory, cont_help, skip_output ) ;
    }

    void close_curve()
    {
        curve_container.back().close_curve();
    }

    void find_intersection(const T& lambda_star, SolutionStorage*& solution_vector)
    {

        for(auto &x: curve_container)
        {
            try
            {
                x.set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
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
    int curve_number = -1;


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & curve_container;
        ar & skip_output;
        ar & curve_number;  //a curve number should be serialized!!!
        //ar & directory;      // should a directory be serialized?               
    }

    
};





}

#endif // __BIFURCATION_DIAGRAM_H__