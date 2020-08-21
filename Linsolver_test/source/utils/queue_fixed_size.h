#ifndef __QUEUE_FIXED_SIZE_H__
#define __QUEUE_FIXED_SIZE_H__

#include <queue>
#include <deque>
#include <iostream>

namespace utils
{

template <typename T, int MaxLen, typename Container=std::deque<T>>
class queue_fixed_size : public std::queue<T, Container> {
public:
    void push(const T& value) 
    {
        if (this->size() == MaxLen) 
        {
           this->c.pop_front();
        }
        std::queue<T, Container>::push(value);
    }
    
    void clear()
    {
            this->c.clear();
    }

    T at(int j)
    {
        return(this->c.at(j));
    }

};

}

#endif // __QUEUE_FIXED_SIZE_H__