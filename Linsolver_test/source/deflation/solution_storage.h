#ifndef __SOLUTIONS_STORAGE_H__
#define __SOLUTIONS_STORAGE_H__


#include <vector>
#include <cmath>

namespace deflation
{

template<class vector_operations>
class solution_storage
{
public:    
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    
    solution_storage(vector_operations*& vec_ops_, int number_of_solutions_, T norm_weight_):
    vec_ops(vec_ops_),
    norm_weight(norm_weight_)
    {
        //reserve elements in container
        //each is of size about 60 bytes => can reserve more =)
        container.reserve(number_of_solutions_);
        vec_ops->init_vector(distance_help); vec_ops->start_use_vector(distance_help);

    }
    ~solution_storage()
    {
        
        container.clear();
        elements_number=0;
        vec_ops->stop_use_vector(distance_help); vec_ops->free_vector(distance_help);

    }

    void push(const T_vec& vect)
    {
    
        container.emplace_back( internal_container(vec_ops, vect) );
        elements_number++;
    
    }

    unsigned int get_size()
    {
        return elements_number;
    }

    void calc_distance(const T_vec& x, T& beta, T_vec& c, int P = 2)
    {
        calc_distance_norms(x, c, P);
        beta = distance;
    }



private:
    T distance = 1;
    T_vec distance_help;
    T norm_weight;

    unsigned int elements_number = 0;
    vector_operations* vec_ops;
    
    void calc_distance_norms(const T_vec& x, T_vec& c, int p)
    {
        T distance_der;

        distance = T(1)/(std::pow(vec_ops->norm(x)/norm_weight,p)*(elements_number+1));
        distance_der = T(p)/(std::pow(vec_ops->norm(x)/norm_weight,p+2)*(elements_number+1));
        //calc: y := mul_x*x
        // c = distance_der*(x-0)
        vec_ops->assign_mul(distance_der/norm_weight, x, c);
        //xxx vec_ops->assign_scalar(0.0, c);
        for(int j=0;j<elements_number;j++)
        {
            //calc: z := mul_x*x + mul_y*y
            //distance_help := x - container[j].get_ref()
            vec_ops->assign_mul(T(1), x, T(-1), container[j].get_ref(), distance_help);

            distance += T(1)/(std::pow(vec_ops->norm(distance_help)/norm_weight,p)*(elements_number+1));
            distance_der = T(p)/(std::pow(vec_ops->norm(distance_help)/norm_weight,p+2)*(elements_number+1));
            //calc: y := mul_x*x + mul_y*y
            //c := c + distance_der*distance_help
            vec_ops->add_mul(distance_der/norm_weight, distance_help, T(1), c);
        }
        distance+=T(1);

    }

    //  XXX
    //  nested class started!
        class internal_container
        {
        public:
            internal_container(vector_operations*& vec_ops_, const T_vec& vec_):
            vec_ops(vec_ops_)
            {
                vec_size = vec_ops->get_vector_size();
                init_array(vec_);         
            }

            //copy constructor
            internal_container(const internal_container& ic_)
            {
                vec_size = ic_.vec_size;
                init_array(ic_.array_);
            }

            //move constructor
            internal_container(internal_container&& ic_):
            vec_size(ic_.vec_size),
            array_(ic_.array_),
            allocated(true),
            owned(true)
            {
                ic_.owned = false;
            }

            ~internal_container()
            {
                if((allocated)&&(owned))
                {
                    vec_ops->stop_use_vector(array_); vec_ops->free_vector(array_);
                }

            }
            // operator overloading and references
            // if located on GPU, then this can be accessed only in a kernel!
            T operator [](size_t i) const
            {
                return array_[i];
            }
            T& operator [](size_t i) 
            {
                return array_[i];
            }
            //reference to the internal storage
            T_vec& get_ref()
            {
                return array_;
            }

        
        private:
            T_vec array_;
            vector_operations* vec_ops;
            size_t vec_size;
            bool allocated = false;
            bool owned = false;

            void init_array(const T_vec& vec_)
            {
                vec_ops->init_vector(array_); vec_ops->start_use_vector(array_);
                allocated = true;
                owned = true;
                vec_ops->assign(vec_, array_);
            }

        };
    //  XXX
    //  nested class stopped!

public:
    //operator overloading
    internal_container operator [](size_t i) const
    {
        return container.at(i);
    }
    internal_container & operator [](size_t i) 
    {
        return container.at(i);
    }    


private:    
    std::vector<internal_container> container;  

};

}


#endif