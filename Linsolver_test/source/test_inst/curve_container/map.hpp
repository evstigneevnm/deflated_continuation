// Example program
#include <iostream>
#include <string>
#include <vector>
#include <map>

int main ()
{
    typedef double T;
    typedef std::map < T, T > container_t;
    typedef std::vector < container_t > data_t;
    
    data_t all_data;
    
    all_data.push_back( {{4.5, -4.5}, {4.55, -2.3}, {4.6, 0.123}, {4.65, 1.12}} );
    
    all_data.push_back( {{4.0,2.34}, {4.1,1.34}, {4.2,0.34}, {4.3,-1.34}, {4.4,-2.34}, {4.5,-3.34}, 
                         {4.6,-4.34}, {4.7,-5.34}, {4.8,-6.34}, {4.9,-7.34}, {5.0,-8.34}, {5.1,-9.34}} );

    
    std::cout << all_data.size () << std::endl;


    for (auto &xxx: all_data)
    {
        for (auto &yyy: xxx)
        {
            std::cout << yyy.second << " ";
        }
        std::cout << std::endl;
        auto upper = xxx.upper_bound(4.572);
        auto lower = xxx.lower_bound(4.572);
        std::cout << (*upper).first << "< * >" << (*lower).first << std::endl;
        
    }


  return (0);
}