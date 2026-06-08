#ifndef __EXCEPTION_FUNCTION_HPP__
#define __EXCEPTION_FUNCTION_HPP__
#include <string>
#include <stdexcept>


bool test_function(int val)
{
    if(val>0)
    {
        printf("\nval > 0\n");
        return(true);
    }
    else
    {
        printf("\nval > 0\n");
        throw std::runtime_error(std::string("test_function -> thrown exception") );
    }

}

bool test_function1(int val)
{
    if(val>0)
    {
        printf("\nval > 0\n");
        return(true);
    }
    else
    {
        printf("\nval > 0\n");
        throw std::runtime_error(std::string("test_function1 -> thrown exception") );
    }

}

#endif // __EXCEPTION_FUNCTION_HPP__