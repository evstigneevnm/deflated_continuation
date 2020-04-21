#ifndef __KNOTS_HPP__
#define __KNOTS_HPP__

#include <string>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>

/**
*	Class that contains knots that are used to perform deflation and continuation check of itersections.
*
*/

namespace container
{
 

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


}

#endif // __KNOTS_HPP__