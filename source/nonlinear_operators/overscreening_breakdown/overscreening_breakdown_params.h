#ifndef __PARAMS_S_OVERSCREENING_BREAKDOWN_H__
#define __PARAMS_S_OVERSCREENING_BREAKDOWN_H__


template<class T>
struct params_s
{
    size_t N = 10;
    T sigma = 1.0;
    T L = 1.0;
    T gamma = 1.0;
    T delta = 1.0;    
    T mu = 1.0;
    T u0 = 1.0;
    int param_number = 0;

    void print_data() const
    {
        std::cout << "=== params_s: " << std::endl;
        std::cout << "=   N = " << N << std::endl;
        std::cout << "=   sigma(0) = " << sigma << std::endl;
        std::cout << "=   L = " << L << std::endl;
        std::cout << "=   gamma(1) = " << gamma << std::endl;
        std::cout << "=   delta(2) = " << delta << std::endl;
        std::cout << "=   mu(3) = " << mu << std::endl;
        std::cout << "=   u0(4) = " << u0 << std::endl;
        std::cout << "=   param_number = " << param_number << std::endl;
        std::cout << "=   .........." << std::endl;
    }
    params_s(size_t N_p, int param_number_p, const std::vector<T>& other_params_p):
    N(N_p),
    param_number(param_number_p) 
    {
        sigma = other_params_p.at(0);
        L = other_params_p.at(1);
        gamma = other_params_p.at(2);
        delta = other_params_p.at(3);
        mu = other_params_p.at(4);
        u0 = other_params_p.at(5);
        print_data();
    }
    params_s() = default;
};

#endif