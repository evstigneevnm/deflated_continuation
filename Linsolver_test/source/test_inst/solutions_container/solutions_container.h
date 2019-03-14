#ifndef __SOLUTIONS_CONTAINER_H__
#define __SOLUTIONS_CONTAINER_H__


#include <vector>

using namespace std;

template<class vector_operations>
class solution_storage
{
public:    
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    
    solution_storage(vector_operations*& vec_ops_, int number_of_solutions_):
    vec_ops(vec_ops_)
    {
        container.reserve(number_of_solutions_);
    }
    ~solution_storage()
    {
        container.clear();
        elements_number=0;
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

private:
    unsigned int elements_number = 0;
    vector_operations* vec_ops;
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
                    vec_ops->free_vector(array_); vec_ops->stop_use_vector(array_);
                }

            }
            //operator overloading and references
            T operator [](size_t i) const
            {
                return array_[i];
            }
            T& operator [](size_t i) 
            {
                return array_[i];
            }
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


private:    
    std::vector<internal_container> container;  

};



#endif