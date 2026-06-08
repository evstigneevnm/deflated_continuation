#ifndef __STABILITY_IRAM_EIGENVALUE_SORTER_H__
#define __STABILITY_IRAM_EIGENVALUE_SORTER_H__ 

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
    

namespace stability
{
namespace detail
{

struct eigenvalue_sorter
{
    
    void set_target_eigs(const std::string& which_)
    {
        
        bool found = false;
        for(auto &p_: sorting_list_permisive)
        {
            if(which_ == p_)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            throw std::logic_error("eigenvalue_sorter::set_target_eigs: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + which_);
        }
        target = which_;
    }
    
    std::string get_target_eigs()const
    {
        return target;
    }
    

    template<class VV>
    void operator()(std::vector< VV >& eigidx_p)const
    {
        if((target == "LM")||(target == "lm"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return std::abs(left_.first) > std::abs(right_.first);
            } 
            );
        }
        else if((target == "LR")||(target == "lr"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return  left_.first.real() > right_.first.real();
            } 
            );
        }
        else if((target == "SR")||(target == "sr"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return  left_.first.real() < right_.first.real();
            } 
            );
        } 
        else
        {
            throw std::logic_error("eigenvalue_sorter: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + target);
        }                
    }
private:
    std::string target;
    std::vector<std::string> sorting_list_permisive = {"LR", "lr", "LM", "lm", "SR", "sr"};
};



template<>
void eigenvalue_sorter::operator()(std::vector< std::complex<double> >& eigidx_p)const 
{
    if((target == "LM")||(target == "lm"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<double>& left_, const  std::complex<double>& right_)
        {
            return std::abs(left_) > std::abs(right_);
        } 
        );
    }
    else if((target == "LR")||(target == "lr"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<double>& left_, const  std::complex<double>& right_)
        {
            return  left_.real() > right_.real();
        } 
        );
    }
    else if((target == "SR")||(target == "sr"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<double>& left_, const  std::complex<double>& right_)
        {
            return  left_.real() < right_.real();
        } 
        );
    } 
    else
    {
        throw std::logic_error("eigenvalue_sorter: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + target);
    }                
}
template<>
void eigenvalue_sorter::operator()(std::vector< std::complex<float> >& eigidx_p)const
{
    if((target == "LM")||(target == "lm"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<float>& left_, const  std::complex<float>& right_)
        {
            return std::abs(left_) > std::abs(right_);
        } 
        );
    }
    else if((target == "LR")||(target == "lr"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<float>& left_, const  std::complex<float>& right_)
        {
            return  left_.real() > right_.real();
        } 
        );
    }
    else if((target == "SR")||(target == "sr"))
    {
        std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const  std::complex<float>& left_, const  std::complex<float>& right_)
        {
            return  left_.real() < right_.real();
        } 
        );
    } 
    else
    {
        throw std::logic_error("eigenvalue_sorter: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + target);
    }                
}

}
}


#endif