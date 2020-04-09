#ifndef __SOLUTIONS_CONTAINER_H__
#define __SOLUTIONS_CONTAINER_H__


#include <vector>
#include <cmath>

template<class vector_operations>
class solution_storage
{
public:    
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    
    solution_storage(vector_operations*& vec_ops_, int number_of_solutions_):
    vec_ops(vec_ops_)
    {
        //reserve elements in container
        //each is of size about 56 bytes => can reserve more =)
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
        container.emplace_back(internal_container(vec_ops, vect));
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

    unsigned int elements_number = 0;
    vector_operations* vec_ops;
    
    void calc_distance_norms(const T_vec& x, T_vec& c, int p)
    {
        T distance_der;

        distance = T(1)/(std::pow(vec_ops->norm(x),p)*(elements_number+1));
        distance_der = T(p)/(std::pow(vec_ops->norm(x),p+2)*(elements_number+1));
        //calc: y := mul_x*x
        // c = distance_der*(x-0)
        vec_ops->assign_mul(distance_der, x, c);
        //xxx vec_ops->assign_scalar(0.0, c);
        for(int j=0;j<elements_number;j++)
        {
            //calc: z := mul_x*x + mul_y*y
            //distance_help := x - container[j].get_ref()
            vec_ops->assign_mul(T(1), x, T(-1), container[j].get_ref(), distance_help);

            distance += T(1)/(std::pow(vec_ops->norm(distance_help),p)*(elements_number+1));
            distance_der = T(p)/(std::pow(vec_ops->norm(distance_help),p+2)*(elements_number+1));
            //calc: y := mul_x*x + mul_y*y
            //c := c + distance_der*distance_help
            vec_ops->add_mul(distance_der, distance_help, T(1), c);
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
    //  nested class stoped!

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
    //begin, end fr c++11 iterator like auto &x: internal_container
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


private:    
    std::vector<internal_container> container;  

};

#endif