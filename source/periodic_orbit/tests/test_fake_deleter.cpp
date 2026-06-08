#include <iostream>
#include <memory>

//g++ source/periodic_orbit/tests/test_fake_deleter.cpp


struct deleter
{
    template<class T>
    void operator()(T* p) const { }
};


class A
{
public:
    A() = default;

};

int main()
{
    double* array = new double[100];
    for(int j = 0; j<100;j++)
    {
        array[j] = j;
        std::cout << array[j] << " ";
    }
    std::cout << std::endl;
    
    std::shared_ptr<double> shared_array;
    shared_array.reset<double, deleter>( array, deleter() );
    
    for(int j = 0; j<100;j++)
        std::cout << shared_array.get()[j] << " ";
    std::cout << std::endl;
    
    delete [] array;
    
    A* a = new A();
    std::shared_ptr<A> a_p;
    a_p.reset<A, deleter>(a, deleter() );
    delete a;

    return 0;
}