#ifndef __CONTINUATION__SOLUTION_POINT_H__
#define __CONTINUATION__SOLUTION_POINT_H__


template<class vector_operations>
class solution_point
{
public:
    typedef typename vector_operations::scalar_type  scalar_type;
    typedef typename vector_operations::vector_type  vector_type;

    solution_point(vector_type &solution_, scalar_type parameter_, const vector_operations *vec_ops_): solution(solution_), parameter(parameter_), vec_ops(vec_ops_)
    {
        own_solution_vector=false;
    }
    
    solution_point(scalar_type solution_value_, scalar_type parameter_, const vector_operations *vec_ops_): parameter(parameter_), vec_ops(vec_ops_)
    {
        vec_ops->init_vector(solution); vec_ops->start_use_vector(solution); 
        own_solution_vector=true;
        vec_ops->assign_scalar(solution_value_, solution);
    }


    ~solution_point()
    {
        if(own_solution_vector){
            vec_ops->stop_use_vector(solution); vec_ops->free_vector(solution);
        }
    }
    
    std::ostream & operator<<(std::ostream & os, const solution_point &sp)
    {
        size_t vec_size = sp.vec_ops->get_vector_size();
        if(sp.vec_ops->device_location()){
            //save to host!?!
        }
        os << sp.parameter << " " << sp.solution_norm_2 << " " << sp.solution_norm_1;
        for(int j=0;j<vec_size-1;j++){
            os << " " << sp.solution[j];
        }
        return os;
    }

protected:
    vector_type solution;
    scalar_type parameter;
    scalar_type solution_norm_2;
    scalar_type solution_norm_1;
    vector_operations *vec_ops;
private:

    bool own_solution_vector=false;

};



#endif