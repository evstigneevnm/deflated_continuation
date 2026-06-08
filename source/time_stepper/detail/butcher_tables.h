#ifndef __TIME_STEPPER_BUTCHER_TABLES_H__
#define __TIME_STEPPER_BUTCHER_TABLES_H__

#include <vector>
#include <cmath>
#include <map>
#include <stdexcept>
#include <limits>
#include <utility>
#include "all_methods_enum.h"


namespace time_steppers
{
namespace detail
{

struct tableu
{
    enum type {ERK, SDIRK, DIRK, IRK};
    using mat_t = std::vector<std::vector< long double> >;
    using vec_t = std::vector< long double>;
    methods method;
    tableu() = default;
    tableu(const methods&& method_p, const mat_t&& A_p, const vec_t&& b_p, const vec_t&& c_p = {}, const vec_t&& b_hat_p = {}):
    method(method_p),
    A_(std::move(A_p)),
    b_(std::move(b_p)),
    c_(std::move(c_p)),
    b_hat_(std::move(b_hat_p)),
    sz_(b_p.size()),
    embeded_method_(b_hat_p.size()>0),
    autonomous_(c_p.size()==0)
    {
        
        check_square_matrix();
        if(embeded_method_)
        {
            err_b_.resize(sz_);
            for(int j=0;j<b_hat_.size();j++)
            {
                err_b_[j] = b_[j] - b_hat_[j];
            }
        }
        if(get_abs_diag() == 0)
        {
            current_type_ = ERK;
        }
        else if( equal_diag() )
        {
            current_type_ = SDIRK;
        }
        else if( get_upper_triang()>0 )
        {
            current_type_ = IRK;
        }
        else
        {
            current_type_ = DIRK;
        }

    }
    tableu(const tableu&) = delete;
    tableu(tableu&& other_p):
    method(other_p.method),
    A_(std::move(other_p.A_)),
    b_(std::move(other_p.b_)),
    c_(std::move(other_p.c_)),
    b_hat_(std::move(other_p.b_hat_)),
    sz_(std::move(other_p.sz_)),
    err_b_(std::move(other_p.err_b_)),
    embeded_method_(std::move(other_p.embeded_method_)),
    autonomous_(std::move(other_p.autonomous_)),
    current_type_(std::move(other_p.current_type_))
    {}

    tableu& operator = (const tableu&) = delete;
    tableu& operator = (tableu&& other_p)
    {
        if(this->A_.size()>0)
        {
            this->A_.clear();
            this->b_.clear();
            this->c_.clear();
            this->b_hat_.clear();
            this->err_b_.clear();
        }
        this->method = std::move(other_p.method);
        this->A_ = std::move(other_p.A_);
        this->b_ = std::move(other_p.b_);
        this->c_= std::move(other_p.c_);
        this->b_hat_ = std::move(other_p.b_hat_);
        this->sz_ = std::move(other_p.sz_);
        this->err_b_ = std::move(other_p.err_b_);
        this->embeded_method_ = std::move(other_p.embeded_method_);
        this->autonomous_ = std::move(other_p.autonomous_);
        this->current_type_ = std::move(other_p.current_type_);
        return *this;
    }

    template<class T>
    T get_A(size_t j, size_t k)const
    {
        return static_cast<T>(A_.at(j).at(k) );
    }
    template<class T>    
    T get_b(size_t k)const
    {
        return static_cast<T>(b_.at(k));
    }
    template<class T>    
    T get_c(size_t j)const
    {
        if(!autonomous_)
            return static_cast<T>(c_.at(j));
        else
            return 0;
    }  
    template<class T>  
    T get_err_b(size_t j)const
    {
        if(embeded_method_)
            return static_cast<T>(err_b_.at(j));
        else
            throw std::logic_error("butcher_tables::tableu: can't obtain error ersimate for a non-embedded method.");
    } 
    bool is_autonomous()const
    {
        return autonomous_;
    }
    bool is_embedded()const
    {
        return embeded_method_;
    }
    type get_type()const
    {
        return current_type_;
    }
    size_t get_size()const
    {
        return sz_;
    }

private:
    size_t sz_;
    mat_t A_; //A_{j,k}:=A[j][k]
    vec_t b_;
    vec_t b_hat_;
    vec_t c_;
    vec_t err_b_;
    bool embeded_method_;
    bool autonomous_;
    type current_type_;

    long double get_abs_diag()const
    {
        long double diag = 0;
        for(size_t j = 0; j<sz_; j++)
        {
            diag += std::abs(A_[j][j]);
        }
        return diag;
    }
    bool equal_diag()const
    {
        auto A00 = A_[0][0];
        bool res = true;
        for(size_t j = 0;j<sz_;j++)
        {
            res = ( std::abs(A00-A_[j][j])>std::numeric_limits<long double>::epsilon() );
            if(!res)
            {
                break;
            }
        }
        return res;
    }

    long double get_upper_triang()const
    {
        long double upper_tri = 0;
        for(size_t j=0;j<sz_;j++)
        {
            for(size_t k=j+1;k<sz_;k++)
            {
                upper_tri += std::abs(A_[j][k]);
            }
        }
        return upper_tri;
    }

    void check_square_matrix()
    {
        size_t n_rows = A_.size();
        for(auto &x: A_)
        {
            if(n_rows != x.size() )
            {
                throw std::logic_error("butcher_tables::tableu: provided A matrix is not square.");
            }
        }
    }

};

// first explicit, then implicit
struct composite_tableu: public tableu
{
    
    composite_tableu():tableu(){}
    composite_tableu(const methods&& method_p, const tableu::mat_t&& A_e, const tableu::mat_t&& A_i, const tableu::vec_t&& b_e, const tableu::vec_t&& b_i, const tableu::vec_t&& c_e = {}, const tableu::vec_t&& c_i = {}, const tableu::vec_t&& b_hat_p_e = {}, const tableu::vec_t&& b_hat_p_i = {}):
    E{std::move(method_p), std::move(A_e), std::move(b_e), std::move(c_e), std::move(b_hat_p_e)},
    I{std::move(method_p), std::move(A_i), std::move(b_i), std::move(c_i), std::move(b_hat_p_i)}
    {

    }

    tableu E, I;


};


struct butcher_tables
{


    butcher_tables()
    {
        tables.emplace("EE",std::move(set_table(EXPLICIT_EULER)));
        tables.emplace("HE",std::move(set_table(HEUN_EULER)));
        tables.emplace("RK33SSP",std::move(set_table(RK33SSP)) );
        tables.emplace("RK43SSP",std::move(set_table(RK43SSP)) ); 
        tables.emplace("RKDP45",std::move(set_table(RKDP45)) ); 
        tables.emplace("RK64SSP",std::move(set_table(RK64SSP)) ); 
        tables.emplace("IE",std::move(set_table(IMPLICIT_EULER)) ); 
        tables.emplace("IM",std::move(set_table(IMPLICIT_MIDPOINT)) ); 
        tables.emplace("CN",std::move(set_table(CRANK_NICOLSON)) ); 
        tables.emplace("SDIRK2A1",std::move(set_table(SDIRK2A1)) ); 
        tables.emplace("ESDIRK3A2",std::move(set_table(ESDIRK3A2)) ); 
        tables.emplace("SDIRK3A3",std::move(set_table(SDIRK3A3)) ); 
    }
    tableu set_table(const methods& method_p) const
    {
        
        switch(method_p)
        {
        case EXPLICIT_EULER:
            return std::move(tableu(EXPLICIT_EULER, {{0}},{1.0},{0}));
            break;  
        case HEUN_EULER:
            return std::move(tableu(HEUN_EULER, {{0,0},{1.0,0}},{0.5,0.5},{0,1.0}, {1.0, 0}));
            break;             
        case RK33SSP:
            return std::move(tableu(RK33SSP, {{0,0,0},{1.0,0,0},{1.0/4.0,1.0/4.0,0}},{1.0/6.0,1.0/6.0,2.0/3.0},{0,1.0,0.5},{0.291485418878409, 0.291485418878409, 0.417029162243181} )); 
            break;
        case RK43SSP:
            return std::move(tableu(RK43SSP, {{0,0,0,0},{0.5,0,0,0},{0.5,0.5,0,0},{1.0/6.0,1.0/6.0,1.0/6.0,0}},{1.0/6.0,1.0/6.0,1.0/6.0,1.0/2.0},{0,0.5,1.0,0.5}, {1.0/3.0, 1.0/3.0, 1.0/3.0, 0} )); 
            //another pair: {0.138870252716866, 0.722259494566267, 0.138870252716866, 0}
            break;
        case RKDP45:
            return std::move(tableu(RKDP45, {{0,0,0,0,0,0,0},{1.0/5.0,0,0,0,0,0,0},{3.0/40.0,9.0/40.0,0,0,0,0,0},{44.0/45.0,-56.0/15.0,32.0/9.0,0,0,0,0},{19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0,0,0,0},{9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0,0,0},{35.0/384.0,0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0,0}},{35.0/384,0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0,0},{0,1.0/5.0,3.0/10.0,4.0/5.0,8.0/9.0,1.0,1.0},{5179.0/57600.0,0,7571.0/16695.0,393.0/640.0,-92097.0/339200.0,187.0/2100.0,1.0/40.0}));
            break;
        
        case RK64SSP:
            return std::move(
                tableu(RK64SSP, 
                    {{0,0,0,0,0,0},{0.3552975516919, 0, 0, 0, 0, 0},{0.2704882223931, 0.3317866983600, 0, 0, 0, 0},{0.1223997401356, 0.1501381660925, 0.1972127376054, 0, 0, 0},{0.0763425067155, 0.0936433683640, 0.1230044665810, 0.2718245927242, 0, 0},{0.0763425067155, 0.0936433683640, 0.1230044665810, 0.2718245927242, 0.4358156542577, 0}},
                    {0.1522491819555, 0.1867521364225, 0.1555370561501, 0.1348455085546, 0.2161974490441, 0.1544186678729},
                    {0,0.3552975516919,0.6022749207532,0.4697506438335,0.5648149343849,1.0006305886426},
                    {0.1210663237182, 0.2308844004550, 0.0853424972752, 0.3450614904457, 0.0305351538213, 0.1871101342844}
                    ));
            break;

        case IMPLICIT_EULER:
            return std::move(tableu(IMPLICIT_EULER, {{1}},{1},{1} ) );
            break;
        case IMPLICIT_MIDPOINT:
            return std::move(tableu(IMPLICIT_MIDPOINT, {{1.0/2.0}},{1.0},{1.0/2.0} ) );
            break;
        case CRANK_NICOLSON:
            return std::move(tableu(CRANK_NICOLSON, {{0,0},{1.0/2.0,1.0/2.0}},{1.0/2.0,1.0/2.0},{0,1.0} ) );
            break;
        case SDIRK2A1:
            {
                double gamma = 0.5*(2.0 - std::sqrt(2.0));
                double b2 = 0.5;
                return std::move(tableu(SDIRK2A1, {{gamma,0},{1.0-gamma, gamma}},{(1.0-gamma),gamma},{gamma,1.0},{(1.0-b2),b2} ) );
            }
            break;            
        case ESDIRK3A2: //DIRK.pdf, p71
            {
                double gamma = 0.5*(2.0 - std::sqrt(2.0));
                double b2 = (gamma*(-2+7*gamma-5*gamma*gamma+4*gamma*gamma*gamma))/(2*(2*gamma-1));
                double b3 = (-2*gamma*gamma*(1-gamma+gamma*gamma))/(2*gamma-1);
                double b1 = 2*gamma;
                return std::move(tableu(ESDIRK3A2, {{0,0,0},{gamma, gamma,0},{(1-b1-gamma), b1, gamma}},{(1.0-b1-gamma), b1, gamma},{0, 2*gamma, 1},{(1-b2-b3),b2,b3} ) );
            }
            break;
        case SDIRK3A3: //DIRK.pdf, p77
            {
                double gamma = 0.43586652150845899941601945;
                double alpha = 1-4*gamma+2*gamma*gamma;
                double beta = -1+6*gamma-9*gamma*gamma+3*gamma*gamma*gamma;
                double b1 = (-1+4*gamma)/(4*beta);
                double b2 = (-3*alpha*alpha)/(4*beta);
                double b3 = 0.5;
                double c2 = (2-9*gamma+6*gamma*gamma)/(3*alpha);
                return std::move(tableu(SDIRK3A3, {{gamma,0,0},{c2-gamma, gamma,0},{(1-b2-gamma), b2, gamma}},{(1.0-b2-gamma), b2, gamma},{gamma, c2, 1},{(1-b2-b3),b2,b3} ) );
            }
            break;

        default:
            throw std::logic_error("butcher_tables: unsupported table enum provided.");
        }   
    }
    
    
    tableu set_table_by_name(const std::string& name)
    {
        auto cc = std::move(tables[name]);
        if(cc.get_size()==0)
        {
            throw std::logic_error("butcher_tables: non-existent table name provided.");
        }
        return std::move(cc);
    }

    std::vector<std::string> get_list_of_table_names() const
    {
        std::vector<std::string> names;
        for(auto &n: tables)
        {
            names.push_back(n.first);
        }
        return names;
    }

private:
    std::map<std::string, tableu> tables;
    

};


struct composite_butcher_tables
{


    composite_butcher_tables()
    {
        composite_tables.emplace("IMEX_EULER", set_table(IMEX_EULER) );
        composite_tables.emplace("IMEX_TR2", set_table(IMEX_TR2) );
        composite_tables.emplace("IMEX_ARS3", set_table(IMEX_ARS3) );
        composite_tables.emplace("IMEX_AS2", set_table(IMEX_AS2) );
    }

    std::pair<tableu, tableu> set_table(const methods& method_p) const
    {
        switch(method_p)
        {
            case IMEX_EULER:
                {
                    auto ct = composite_tableu(IMEX_EULER, {{0,0},{1,0}}, {{0,0},{0,1}}, {1, 0},{0, 1},{0, 1},{0, 1});
                    return std::pair<tableu,tableu>( std::move(ct.E), std::move(ct.I)); 
                }
                break;  
            case IMEX_TR2:
                {
                    auto ct = composite_tableu(IMEX_TR2, {{0,0},{1,0}}, {{0,0},{1.0/2.0,1.0/2.0}}, {1.0/2.0, 1.0/2.0},{1.0/2.0, 1.0/2.0},{0, 1},{0, 1});
                    return std::pair<tableu,tableu>( std::move(ct.E), std::move(ct.I)); 
                }
                break; 
            case IMEX_ARS3:
                {
                    double gamma = (3.0+std::sqrt(3.0))/6.0;
                    auto ct = composite_tableu(IMEX_ARS3, {{0,0,0},{gamma,0,0},{gamma-1,2*(1-gamma),0}}, {{0,0,0},{0,gamma,0},{0, 1-2*gamma, gamma}}, {0, 1.0/2.0, 1.0/2.0},{0, 1.0/2.0, 1.0/2.0},{0, gamma, 1-gamma},{0, gamma, 1-gamma});
                    return std::pair<tableu,tableu>( std::move(ct.E), std::move(ct.I)); 
                }
                break;                                 
            case IMEX_AS2:
                {
                    double omega = (2.0-std::sqrt(2.0))/2.0;
                    double k = 1 - 1.0/(2.0*omega);
                    auto ct = composite_tableu(IMEX_AS2, {{0,0,0},{omega,0,0},{k,1-k,0}}, {{0,0,0},{0,omega,0},{0, 1-omega, omega}}, {k, 1-k, 0},{0, 1-omega, omega},{0, omega, 1},{0, omega, 1});
                    return std::pair<tableu,tableu>( std::move(ct.E), std::move(ct.I)); 
                }
                break;  
            default:
                  
                throw std::logic_error("composite_butcher_tables: unsupported table enum provided.");                      
        }        


    }


    std::pair<tableu, tableu> set_table_by_name(const std::string& name)
    {
        auto cc = std::move(composite_tables[name]);
        if(cc.first.get_size()==0)
        {
            throw std::logic_error("composite_butcher_tables: non-existent table name provided.");
        }
        return std::move(cc);
    }


    std::vector<std::string> get_list_of_table_names() const
    {
        std::vector<std::string> names;
        for(auto &n: composite_tables)
        {
            names.push_back(n.first);
        }
        return names;
    }


private:
    std::map< std::string, std::pair<tableu, tableu> > composite_tables;

};

}

std::string get_scheme_type_by_name(const std::string& name)
{
    detail::butcher_tables bt;
    detail::composite_butcher_tables ct;  
    // ERK, SDIRK, DIRK, IRK
    std::string ret_type = "";
    bool found = false;
    using table_type = detail::tableu::type;
    for(auto &v: bt.get_list_of_table_names() )
    {
        if(v == name)
        {
            found = true;
            auto table = bt.set_table_by_name(v);
            if( table.get_type() == table_type::ERK )
            {
                ret_type = "explicit";
            }
            else
            {
                ret_type = "implicit";
            }
            break;
        }
    }
    for( auto &v: ct.get_list_of_table_names() )
    {
        if(v == name)
        {
            found = true;
            ret_type = "imex";
        }
    }
    if(!found)
    {
        throw std::runtime_error("provided name " + name + " is not found in tables, no such schemes.");
    }
    return ret_type;
}

}



#endif