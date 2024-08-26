#ifndef __SOLUTIONS_STORAGE_H__
#define __SOLUTIONS_STORAGE_H__


#include <vector>
#include <cmath>
#include <iostream>

namespace deflation
{

template<class VectorOperations>
class solution_storage
{
public:    
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    
    solution_storage(VectorOperations* vec_ops_, unsigned int number_of_solutions, T norm_weight_, T P_ = T(2.0)):
    vec_ops(vec_ops_),
    norm_weight(norm_weight_),
    P(P_),
    ignore_zero_(false),
    number_of_solutions_(number_of_solutions)
    {
        //reserve elements in container
        //each is of size about 60 bytes => can reserve more =)
        
        container.reserve(number_of_solutions);
        vec_ops->init_vector(distance_help); vec_ops->start_use_vector(distance_help);
        vec_ops->init_vector(x0_); vec_ops->start_use_vector(x0_);
        vec_ops->assign_scalar(static_cast<T>(0.0), x0_);
        // norm_weight = vec_ops->get_l2_size(); //overide norm weight due to norm_l2 operator in vector operations.
        vec_ops->init_vector(distance_help_translate); vec_ops->start_use_vector(distance_help_translate);

    }
    ~solution_storage()
    {
        
        container.clear();
        elements_number = 0;
        vec_ops->stop_use_vector(x0_); vec_ops->free_vector(x0_);
        vec_ops->stop_use_vector(distance_help_translate); vec_ops->free_vector(distance_help_translate);
        vec_ops->stop_use_vector(distance_help); vec_ops->free_vector(distance_help);


    }

    void set_known_solution(const T_vec& x0_p)
    {
        vec_ops->assign(x0_p, x0_);
        ignore_zero_ = false;
    }    
    
    void set_ignore_zero()
    {
        ignore_zero_ = true;
    }

    void push_back(const T_vec& vect)
    {
    
        container.emplace_back(vec_ops, vect);
        elements_number++;
    
    }
    void clear()
    {
        container.clear();
        elements_number = 0;
    }

    unsigned int get_size()
    {
        return elements_number;
    }

    void calc_distance(const T_vec& x, T& beta, T_vec& c)
    {
        calc_distance_norms(x, c, P);
        beta = distance;
    }


    void calc_distance(const T_vec& x, const T_vec& x_translate, T& beta, T_vec& c)
    {
        calc_distance_norms(x, x_translate, c, P);
        beta = distance;
    }

    unsigned int get_number_of_reserved_solutions() const
    {
        return number_of_solutions_;
    }


private:
    T distance = 1;
    T_vec distance_help;
    T_vec distance_help_translate;
    T_vec x0_;
    T norm_weight;
    T P; //power of the distance
    bool ignore_zero_;
    unsigned int number_of_solutions_;

    unsigned int elements_number = 0;
    VectorOperations* vec_ops;
    
    void calc_distance_norms(const T_vec& x, const T_vec& x_translate, T_vec& c, const T p)
    {
        T distance_der;

        vec_ops->assign_mul(static_cast<T>(1.0), x_translate, static_cast<T>(-1.0), x0_, distance_help);
        distance = T(1)/(pow(vec_ops->norm_l2(distance_help),p)*(elements_number+1)); //distance to zero
        if(ignore_zero_)
            distance = 0;

        distance_der = T(p)/(pow(vec_ops->norm_l2(distance_help), p+T(2.0))*(elements_number+1)); //distance to zero
        if(ignore_zero_)
            distance_der = 0;        
        //calc: y := mul_x*x
        // c = distance_der*(x-0)
        vec_ops->assign_mul(distance_der/norm_weight, x, c);
        //xxx vec_ops->assign_scalar(0.0, c);
        for(int j=0;j<elements_number;j++)
        {
            //calc: z := mul_x*x + mul_y*y
            //distance_help := x - container[j].get_ref()
            // vec_ops->assign_mul(T(1), x, T(-1), container[j].get_ref(), distance_help);
            vec_ops->assign_mul(T(1), x_translate, T(-1), container[j].get_ref(), distance_help);
            // vec_ops->assign_mul(T(1), x_translate, T(-1), container[j].get_ref(), distance_help_translate);

            distance += T(1)/(pow(vec_ops->norm_l2(distance_help),p)*(elements_number+1));
            distance_der = T(p)/(pow(vec_ops->norm_l2(distance_help),p+T(2.0))*(elements_number+1));
            //calc: y := mul_x*x + mul_y*y
            //c := c + distance_der*distance_help
            vec_ops->add_mul(distance_der/norm_weight, distance_help, T(1), c);
        }
        distance+=T(1);

    }

    void calc_distance_norms(const T_vec& x, T_vec& c, const T p)
    {
        T distance_der;
        //M=(1./(S.*(w.*normE(x_s-x0_s(:,1))).^p));
        //V=(-p./(S.*(w.*normE(x_s-x0_s(:,1))).^(p+2))).*w.*(x_s-x0_s(:,1))';
        vec_ops->assign_mul(static_cast<T>(1.0), x, static_cast<T>(-1.0), x0_, distance_help);
        auto total_elements = elements_number+1;
        distance = static_cast<T>(1.0)/(pow(vec_ops->norm_l2(distance_help),p)*(total_elements)); //distance to zero
        if(ignore_zero_)
        {
            distance = 0;
            total_elements = elements_number;
        }
        distance_der = static_cast<T>(p)/(pow(vec_ops->norm_l2(distance_help), p+T(2.0))*(total_elements)); //distance to zero
        if(ignore_zero_)
        {
            distance_der = 0;        
        }
        //calc: y := mul_x*x
        // c = distance_der*(x-0)
        vec_ops->assign_mul(distance_der/norm_weight, x, c);
        //xxx vec_ops->assign_scalar(0.0, c);
        for(int j=0;j<elements_number;j++)
        {
            
            //calc: z := mul_x*x + mul_y*y
            //distance_help := x - container[j].get_ref()
            vec_ops->assign_mul(static_cast<T>(1.0), x, static_cast<T>(-1.0), container[j].get_ref(), distance_help);

            distance += static_cast<T>(1.0)/(pow(vec_ops->norm_l2(distance_help),p)*(total_elements));
            distance_der = static_cast<T>(p)/(pow(vec_ops->norm_l2(distance_help),p+T(2.0))*(total_elements));
            //calc: y := mul_x*x + mul_y*y
            //c := c + distance_der*distance_help
            vec_ops->add_mul(distance_der/norm_weight, distance_help, static_cast<T>(1.0), c);
        }
        distance+=static_cast<T>(1.0);

    }


    //  XXX
    //  nested class started!
        class internal_container
        {
        public:
            internal_container(VectorOperations* vec_ops_, const T_vec& vec_):
            vec_ops(vec_ops_)
            {
                vec_size = vec_ops->get_vector_size();
                init_array(vec_);         
            }

            //copy constructor
            internal_container(const internal_container& ic_)
            {
                vec_ops = ic_.vec_ops;
                vec_size = ic_.vec_size;
                init_array(ic_.array_);
            }

            //move constructor
            internal_container(internal_container&& ic_):
            vec_ops(ic_.vec_ops),
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
                    std::cout << "removed from container" << std::endl;
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
            void copy(T_vec& vec_x)
            {
                if((owned)&&(allocated))
                {
                    std::cout << "extern:" << vec_ops->check_is_valid_number(vec_x) << std::endl;
                    std::cout << "intern:" << vec_ops->check_is_valid_number(array_) << std::endl;
                    std::fflush(stdout);
                    vec_ops->assign(array_, vec_x);
                }
            }
        
        private:
            T_vec array_;
            VectorOperations* vec_ops;
            size_t vec_size;
            bool allocated = false;
            bool owned = false;

            void init_array(const T_vec& vec_)
            {
                vec_ops->init_vector(array_); vec_ops->start_use_vector(array_);
                vec_ops->assign(vec_, array_);
                allocated = true;
                owned = true;
                
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
    //define types in order to use iterators
    typedef typename std::vector<internal_container> container_t;
    container_t container;  
    typedef typename container_t::iterator iterator_t;
    typedef typename container_t::const_iterator const_iterator_t;

public:
    //recast iterators from std::vector<internal_container> to use in for(auto& x:...)
    inline iterator_t begin() noexcept 
    { 
        return container.begin(); 
    }
    inline const_iterator_t cbegin() const noexcept 
    { 
        return container.cbegin(); 
    }
    inline iterator_t end() noexcept 
    { 
        return container.end(); 
    }
    inline const_iterator_t cend() const noexcept 
    { 
        return container.cend(); 
    }

};

}


#endif