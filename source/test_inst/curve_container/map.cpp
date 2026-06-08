// Example program
#include <iostream>
#include <string>
#include <vector>
#include <map>






int main ()
{
    uint64_t aaa = 0;

    typedef double T;
    typedef std::multimap < T, T > container_t;
    typedef std::vector < container_t > data_t;
    
    data_t all_data;
    
    all_data.push_back( {{4.5, -3.5}, {4.55, -2.3}, {4.6, 0.123}, {4.65, 1.12},
                          {4.66, 1.15}, {4.61, 1.89}, {4.6, 1.95}, {4.5, 2.22}, {4.4, 3.02}, {4.3, 3.8},
                          {4.4, 4.15}, {4.5, 4.89}, {4.6, 4.95}, {4.7, 5.22}, {4.8, 6.02}, {4.9, 6.8}} );
    
    all_data.push_back( {{4.0,2.34}, {4.1,1.34}, {4.2,0.34}, {4.3,-1.34}, {4.4,-2.34}, {4.5,-3.34}, 
                         {4.6,-4.34}, {4.7,-5.34}, {4.8,-6.34}, {4.9,-7.34}, {5.0,-8.34}, {5.1,-9.34}} );

    
    // container_t bbb;
    // unsigned int N = 1000000;
    // for(int j=0;j<N;j++)
    // {

    //     bbb.insert({j,N/2-j});
    //     if(N%10000 == 0)
    //         std::cout << 100.0*j/N << "% \r";
    // }




    std::cout << all_data.size () << std::endl;


    for (auto &xxx: all_data)
    {
        for (auto &yyy: xxx)
        {
            std::cout << yyy.first << ":" << yyy.second << " ";
        }
        std::cout << std::endl;
        auto lower = xxx.lower_bound(4.55);
        if(lower!=xxx.end())
        {
            T val = (*lower).first;
            auto all_found = xxx.equal_range(val);
            for(auto &x = all_found.first; x!=all_found.second; x++)
            {
                std::cout << x->first << " " << x->second <<  std::endl;
            }
        }
        std::cout << "-------------------------" << std::endl << std::endl;
    }


  return (0);
}