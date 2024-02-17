#ifndef __SCFD_DETAIL_DENSE_OPERATIONS_H__
#define __SCFD_DETAIL_DENSE_OPERATIONS_H__

#include <cmath>
#include <utility>
#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <memory>
#include <random>

#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

namespace numerical_algos
{
namespace lin_solvers 
{
namespace detail
{
template<class Card, class Type, class DenseVectorType = scfd::arrays::tensor0_array_nd<Type, 1, scfd::memory::host>, class DenseMatrixType = scfd::arrays::tensor0_array_nd<Type, 2, scfd::memory::host> >
class dense_operations
{
private:
    using T = Type;
    mutable Card rows_ = 0, cols_ = 0;

    void check_init() const
    {
        if((rows_ == 0)&&(cols_ == 0))
        {

            throw std::logic_error("numerical_algos:lin_solvers:detail:dense_operations:: methods called withought class delayed initialization.");
        }
    }

public:
    using scalar_type = Type;
    using vector_type = DenseVectorType;
    using matrix_type = DenseMatrixType;

private:
    std::random_device rd;
    mutable std::mt19937 gen;
    mutable std::uniform_real_distribution<> dis;

public:

    dense_operations(Card rows, Card cols):
    rows_(rows), cols_(cols)
    {
        common_constructor_operations();
    }

    dense_operations():
    rows_(0), cols_(0)
    {
        common_constructor_operations();
    }

    void common_constructor_operations()
    {
        gen =  std::mt19937(rd());
        dis = std::uniform_real_distribution<>(-1.0, 1.0);
    }

    ~dense_operations() = default;

    void init(Card rows, Card cols) const
    {
        rows_ = rows;
        cols_ = cols;
    }


    void init_row_vector(vector_type& vec) const
    {
        check_init();
        vec.init(cols_);
    }
    template<class...Args>    
    void init_row_vectors(Args&&...args) const
    {
        check_init();
        std::initializer_list<int>{((void)init_row_vector(std::forward<Args>(args)), 0)...};
    }

    void init_col_vector(vector_type& vec) const
    {
        check_init();   
        vec.init(rows_);
    }
    template<class...Args>    
    void init_col_vectors(Args&&...args) const
    {
        check_init();
        std::initializer_list<int>{((void)init_col_vector(std::forward<Args>(args)), 0)...};
    }
    // template<class...Args>     //c++17
    // void init_col_vectors(Args&&...args) const
    // {
    //     (init_col_vector(std::forward<Args>(args)),...);
    // }  

    void init_matrix(matrix_type& mat) const
    {
        check_init();
        mat.init(rows_, cols_);
    }
    template<class...Args>    
    void init_matrices(Args&&...args) const
    {
        check_init();
        std::initializer_list<int>{((void)init_matrix(std::forward<Args>(args)), 0)...};
    }    
    void free_row_vector(vector_type& vec) const
    {
        vec.free();
    }
    template<class...Args>    
    void free_row_vectors(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_row_vector(std::forward<Args>(args)), 0)...};
    }    
    void free_col_vector(vector_type& vec) const
    {
        vec.free();
    }
    template<class...Args>    
    void free_col_vectors(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_col_vector(std::forward<Args>(args)), 0)...};
    }     
    void free_matrix(matrix_type& mat) const
    {
        mat.free();
    }
    template<class...Args>     
    void free_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_matrix(std::forward<Args>(args)), 0)...};
    }       
    std::pair<Card, Card> size() const
    {
        return {rows_, cols_};
    }
    bool is_valid_row_vector(const vector_type& vec) const
    {
        check_init();
        bool res = true;
        for(Card j=0;j<cols_;j++)
        {
            if(!std::isfinite(vec(j)))
            {
                res = false;
                break;
            }
        }
        return res;
    }
    bool is_valid_col_vector(const vector_type& vec) const
    {
        check_init();
        bool res = true;
        for(Card j=0;j<rows_;j++)
        {
            if(!std::isfinite(vec(j)))
            {
                res = false;
                break;
            }
        }
        return res;
    }
    bool is_valid_matrix(const matrix_type& mat) const
    {
        check_init();
        bool res = true;
        for(Card j=0;j<rows_;j++)
        {
            for(Card k=0;k<cols_;k++)
            {
                if(!std::isfinite(mat(j,k)))
                {
                    res = false;
                    break;
                }
            }
            if(!res)
            {
                break;
            }
        }
        return res;
    }
    void assign_scalar_row_vector(const scalar_type scalar, vector_type& vec) const
    {
        check_init();
        for(Card j=0;j<cols_;j++)
        {
            vec(j) = scalar;
        }
    }
    void assign_scalar_col_vector(const scalar_type scalar, vector_type& vec)
    {
        check_init();
        for(Card j=0;j<rows_;j++)
        {
            vec(j) = scalar;
        }        
    }
    void assign_scalar_matrix(const scalar_type scalar, matrix_type& mat) const
    {
        check_init();
        for(Card j=0;j<rows_;j++)
        {
            for(Card k=0;k<cols_;k++)
            {
                mat(j,k) = scalar;
            }
        }        
    }    
    void assign_row_vector(const vector_type& x, vector_type& y) const
    {
        check_init();
        for(Card j=0;j<cols_;j++)
        {
            y(j) = x(j);
        }        
    }
    void assign_col_vector(const vector_type& x, vector_type& y) const
    {
        check_init();
        for(Card j=0;j<rows_;j++)
        {
            y(j) = x(j);
        }        
    }
    void assign_matrix(const matrix_type& A, matrix_type& B) const
    {
        check_init();
        for(Card j=0;j<rows_;j++)
        {
            for(Card k=0;k<cols_;k++)
            {
                B(j,k) = A(j,k);
            }        
        }
    }   
    Type& matrix_at(const matrix_type& A, const Card row, const Card col)
    {
        return A(row, col);
    }
    Type& vector_at(const vector_type& x,  const Card j)
    {
        return x(j);
    }

    void matrix_set_column(const vector_type& col_vec, const Card col, matrix_type& mat) const
    {
        check_init();
        for(Card j = 0;j<rows_;j++)
        {
            mat(j,col) = col_vec(j);
        }
    }
    void matrix_set_row(const vector_type& row_vec, const Card row, matrix_type& mat) const
    {
        check_init();
        for(Card j = 0;j<cols_;j++)
        {
            mat(row, j) = row_vec(j);
        }
    }


    void set_random_row_vector(vector_type& vec, const T& from = 0, const T& to = 1) const 
    {
        for(Card j=0;j<rows_;j++)
        {
            T val = dis(gen); // map [-1,1] -> [from,to]
            val = (to-from)*(0.5*(val+1))+from;
            vec(j) = val;
        }
    }
    void set_random_col_vector(vector_type& vec, const T& from = 0, const T& to = 1) const 
    {
        for(Card j=0;j<cols_;j++)
        {
            T val = dis(gen); // map [-1,1] -> [from,to]
            val = (to-from)*(0.5*(val+1))+from;
            vec(j) = val;
        }
    }
    void set_random_matrix(matrix_type& mat, const T& from = 0, const T& to = 1) const
    {
        for(Card j=0;j<rows_;j++)
        {
            for(Card k=0;k<cols_;k++)
            {
                T val = dis(gen); // map [-1,1] -> [from,to]
                val = (to-from)*(0.5*(val+1))+from;
                mat(j,k) = val;
            }        
        }        
    }

    T norm_col_vector(const vector_type& vec) const
    {
        T norm = 0;
        for(Card j=0;j<cols_;j++)
        {
            norm += vec(j)*vec(j);
        }  
        return std::sqrt(norm);
    }
    
    T norm_row_vector(const vector_type& vec) const
    {
        T norm = 0;
        for(Card j=0;j<rows_;j++)
        {
            norm += vec(j)*vec(j);
        }  
        return std::sqrt(norm);   
    }

    void solve_upper_triangular_subsystem(const matrix_type& A, vector_type& x, const Card ind) const 
    {
        for(Card j = ind; j-->0;)
        {   
            x(j) /= A(j,j);
            for (Card k = 0; k < j; ++k)
            {
                x(k) -= A(k, j)*x(j);
            }
        }
    }
    void solve_upper_triangular_subsystem(const matrix_type& A, const vector_type& b, vector_type& x, const Card ind) const    
    {
        if(ind>0)
        {
            assign_col_vector(b, x);
            solve_upper_triangular_subsystem(A, x, ind);
        }
    } 

    void apply_plane_rotation(scalar_type& dx, scalar_type& dy, const scalar_type& cs, const scalar_type& sn) const
    {
        T temp = cs * dx + sn * dy;
        dy = -sn * dx + cs * dy; //sn->conj(sn)
        dx = temp;        
    }
    void generate_plane_rotation(const scalar_type& dx, const scalar_type& dy, scalar_type& cs, scalar_type& sn) const
    {
        if (dy == static_cast<T>(0))
        {
            cs = static_cast<T>(1);
            sn = static_cast<T>(0);
        }
        // else
        // {
        //     //TODO: norm type problem again!
        //     T scale = std::abs(dx) + std::abs(dy);
        //     T norm = scale * std::sqrt(std::abs(dx / scale) * std::abs(dx / scale) +
        //                                       std::abs(dy / scale) * std::abs(dy / scale));
        //     T alpha = dx / std::abs(dx);
        //     cs = std::abs(dx) / norm;
        //     sn = alpha * conj_(dy) / norm;
        // }
        else if(std::abs(dy) > std::abs(dx))
        {
            T tmp = dx / dy;
            sn = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + tmp*tmp);
            cs = tmp*sn;
        } 
        else
        {   
            T tmp = dy / dx;
            cs = static_cast<T>(1) / std::sqrt(static_cast<T>(1) + tmp*tmp);
            sn = tmp*cs;
        }    
    }
    void plane_rotation_col(matrix_type& H, vector_type& cs_, vector_type& sn_, vector_type& s, const Card col) const
    {
        for (int k = 0; k < col; k++)
        {
            apply_plane_rotation(H(k, col), H(k+1, col), cs_(k), sn_(k) );
        }

        generate_plane_rotation(H(col, col), H(col+1, col), cs_(col), sn_(col) );
        apply_plane_rotation(H(col,col), H(col+1,col), cs_(col), sn_(col));
        H(col+1,col) = static_cast<T>(0); //remove numerical noise below diagonal
        apply_plane_rotation(s(col), s(col+1), cs_(col), sn_(col) );
    }
    
    void print_col_vector(const vector_type& vec, int prec = 2)
    {
        if(prec > 2)
            std::cout << std::setprecision(prec) << std::scientific;
        else
            std::cout << std::setprecision(2) << std::fixed;
        for(Card j=0;j<rows_;j++)
        {
            std::cout <<  vec(j) << std::endl;
        }
    }
    void print_row_vector(const vector_type& vec, int prec = 2)
    {
        if(prec > 2)
            std::cout << std::setprecision(prec) << std::scientific;
        else
            std::cout << std::setprecision(2) << std::fixed;

        for(Card j=0;j<cols_;j++)
        {
            std::cout << vec(j) << " ";
        }
    }
    void print_matrix(const matrix_type& H, int prec = 2)
    {
        {
        if(prec > 2)
            std::cout << std::setprecision(prec) << std::scientific;
        else
            std::cout << std::setprecision(2) << std::fixed;

        for(Card j=0;j<rows_;j++)
        {
            for(Card k = 0;k<cols_;k++)
            {
                std::cout <<  H(j,k) << " ";
            }
            std::cout << std::endl;
        }
        }
    }

};

}
}
}

#endif

