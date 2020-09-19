#include <iostream>
#include <string>
#include <vector>

template<class T>
struct external
{
    struct internal_s
    {
        T x;
        T y;
        void set_default()
        {
            x = 0.0;
            y = 0.0;
        }        
    };
    
    
    T x;
    T y;
    internal_s internal;  

    void set_default()
    {
        x = 0.0;
        y = 0.0;
        internal.set_default();
    }
};



int main()
{
    
    external<double> ex = {1.0, 2.0, {-1.0, -2.0}};
    ex.set_default();

    std::cout << ex.x << " " << ex.y << " " << std::endl << ex.internal.x << " " << ex.internal.y << " " << std::endl;
    
    return 0;
}