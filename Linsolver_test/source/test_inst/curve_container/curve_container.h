#ifndef __CURVE_CONTAINER_H__
#define __CURVE_CONTAINER_H__

#include <string>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>



template<class T>
struct complex_key
{
    T key1;
    T key2;

};

template<class key_pare>
struct compare_key_pare
{
   bool operator() (const key_pare& lhs, const key_pare& rhs) const
   {    
       
        bool key1_check = lhs.key1 < rhs.key1;
        if(key1_check)
        {
            return key1_check;
        }
        else if(lhs.key1 == rhs.key1)
        {
            return lhs.key2 < rhs.key2;
        }
        else 
        {
            return key1_check; 
        }
   }

};


template<class T> 
class knots  //contains knots at which deflation is considered
{
private:
    typedef typename std::vector<T> container_t;
    container_t container;  
    typedef typename container_t::iterator iterator_t;
    typedef typename container_t::const_iterator const_iterator_t;
    unsigned int current_position = 0;

    void unique_and_sort()
    {
        std::sort(container.begin(), container.end());
        container.erase( std::unique( container.begin(), container.end() ), container.end() );
    }
public:
// public:
//     knots();

//     ~knots();
    void add_element(T value_)
    {
        container.push_back(value_);
        unique_and_sort();
    }

    void add_element(container_t vec_)
    {
        container.insert(container.end(), std::make_move_iterator(vec_.begin()), std::make_move_iterator(vec_.end()) );
        
        unique_and_sort();
    }
    
    T get_max_value()
    {
        return container.back();
    }

    T get_min_value()
    {
        return container.front();
    }
    T get_value()
    {
        return container[current_position];
    }

    bool next()
    {
        current_position++;
        if( current_position >= size())
        {
            current_position = 0;
            return(false);
        }
        else
            return(true);
    }

    iterator_t next_iterator()
    {
        auto vi = begin();
        std::advance(vi, current_position);
        next();
        return(vi);
    }

    int size()
    {
        return(container.size());
    }


    T operator [](size_t i) const
    {
        return container.at(i);
    }
    T & operator [](size_t i) 
    {
        return container.at(i);
    }  
    //begin, end for c++11 iterator like "auto &x: keys ..."
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


template<class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class Knots>
class curve_storage
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    
    typedef complex_key<T> key_t;
    typedef compare_key_pare<key_t> cmp_keys_t;
    typedef std::map< key_t, T_vec, cmp_keys_t > curve_t; // key_type, value_type, key_comparison_structure

    curve_storage(VectorOperations* vec_ops_p_, Loggin* log_, NewtonMethod* newton_method_, NonlinearOperator* nonlinear_operator_, Knots* knots_, T eps_ = T(1.0e-9) ):
    vec_ops_p(vec_ops_p_),
    log(log_), 
    newton(newton_method_),
    nonlinear_operator(nonlinear_operator_),
    knots(knots_),
    epsilon(eps_)
    {
        log->info("curve_storage::constructor.");
    }
    ~curve_storage()
    {
        log->info("curve_storage::distructor.");
    }

    void get_knots()
    {
        lambda_min = knots->get_min_value();
        lambda_max = knots->get_max_value();  
        // lambda1 =  lambda_min;
        // lambda0 = lambda_min;
    }

    bool interpolate_two_solutions(const T& lambda1, T_vec& x1)
    {
        bool converged;
        converged = newton->solve(nonlinear_operator, x1, lambda1);
        return(converged);
    }


    bool activate_curve()
    {
        active_curve = true;
        lambda_start = current_knot();
        return active_curve;
    }

    bool diactivate_curve()
    {
        active_curve = false;
        return active_curve;
    }

    bool next_knot()
    {
        return( knots->next() );
    }

    T current_knot()
    {
        return( knots->get_value() );
    }

    bool add(const T& lambda_p, const T_vec& x_p)
    {
        log->info("curve_storage::add");
        if(active_curve)
        {
            if(check_lambda(lambda_p))
            {

                //curve_object.insert({DT_, Hist(DT_, unit_, 1)});    
                return(true);
            }
            else
            {
                
                return(false);
            }
            

        }
        else
        {
            //log->error("curve_storage::add: curve is diactivated!\n");
            throw std::runtime_error(std::string("curve_storage::add: adding to diactivated curve!") );
        }
        
    }


private:
    VectorOperations* vec_ops_p;
    Loggin* log;
    NewtonMethod* newton; 
    Knots* knots;
    NonlinearOperator* nonlinear_operator;

    T epsilon;
    T lambda0, lambda_start;

    size_t maximum_ram_size;
    curve_t curve_object;
    unsigned int current_curve_number = 0;
    T lambda_min, lambda_max;
    bool active_curve = false;

    bool check_lambda(T lambda_p)
    {
        bool flag = true;
        if( (lambda0 - lambda_start)*(lambda_p - lambda_start)<T(0) )
        {

            flag = false;
        }
        else
        {
            lambda0 = lambda_p;
        }
        return(flag);

    }

    void generate_curve_number()
    {
        current_curve_number++;
    }


        


};


#endif