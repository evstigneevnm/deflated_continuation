#ifndef __SCFD_LINSPACE_HYPRE_OPERATIONS_H__
#define __SCFD_LINSPACE_HYPRE_OPERATIONS_H__

#include <memory>
#include <vector>
#include <ctime>
#include <vector>
#include <mpi.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <scfd/communication/mpi_comm_info.h>

#include "vector_operations_base.h"
#include <common/hypre_safe_call.h>
#include <common/hypre_matrix.h>
#include <common/hypre_vector.h>
#include <profiling.h>
#include <current_for_each.h>

#include "_hypre_utilities.hpp"

namespace scfd
{
namespace linspace
{

namespace detail
{
namespace kernel
{

template<class Ord, class T, class Vec>
struct mul_pointwise
{
    T mul_;
    Vec in_;
    Vec out_;
    __DEVICE_TAG__ void operator()(Ord j)const
    {
        out_[j] *= mul_*in_[j];
    }
};



}

}

class hypre_vector_space;

struct hypre_operations
{

    using matrix_type = scfd::linspace::hypre_matrix<>;
    using vector_type = scfd::linspace::hypre_vector<>;
    using scalar_type = typename matrix_type::scalar_type; 
    using ordinal_type = typename matrix_type::ordinal_type;
    using big_ordinal_type = typename matrix_type::big_ordinal_type;
    using multivector_type = typename std::vector<vector_type>;
    using vector_space_type = hypre_vector_space;
    using comm_info_type = scfd::communication::mpi_comm_info;
    comm_info_type mpi_data_;

    using for_each_t = current_for_each_1d<ordinal_type>;
    for_each_t for_each;

    hypre_operations(const comm_info_type& mpi_p):
    mpi_data_(mpi_p)
    {}
    
    //x->y
    /*--------------------------------------------------------------------------
     * hypre_SeqVectorCopy
     * copies data from x to y
     * if size of x is larger than y only the first size_y elements of x are
     * copied to y
     *--------------------------------------------------------------------------*/    
    void assign(const vector_type& x, vector_type& y) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign");
        HYPRE_SAFE_CALL( HYPRE_ParVectorCopy(x.data(), y.data() ) );
    }

    void assign_scalar(const scalar_type val, vector_type& x)const 
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign_scalar");
        HYPRE_SAFE_CALL(HYPRE_ParVectorSetConstantValues(x.data(), val));
    }
    // x<-mul_x*x+scalar
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::add_mul_scalar");
        auto x_loc = hypre_ParVectorLocalVector( x.data() );
        auto sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 

        // auto p = ::thrust::device_pointer_cast( hypre_VectorData(x_loc) );
        // ::thrust::transform(p, p+sz_loc, p, add_mul_scalar_functor(scalar, mul_x) );
        HYPRE_THRUST_CALL(transform, x_loc_ptr, x_loc_ptr+sz_loc, x_loc_ptr, add_mul_scalar_functor(scalar, mul_x) );
    }

    void assign_random(vector_type& x)const 
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign_random");
        ordinal_type seed_0 = std::time(0);
        HYPRE_SAFE_CALL(HYPRE_ParVectorSetRandomValues(x.data(), seed_0));
    }

    void scale(const scalar_type scale, vector_type& x)
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::scale");
        HYPRE_SAFE_CALL(HYPRE_ParVectorScale(scale, x.data()));
    }
    [[nodiscard]] scalar_type sum(const vector_type& x)const 
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::sum");
        
        // auto mem_loc = hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(x.data()));
        // std::cout << "mem_loc = " << mem_loc << std::endl; 

        auto x_loc = hypre_ParVectorLocalVector( x.data() );
        auto sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 
        auto res_loc = HYPRE_THRUST_CALL(reduce, x_loc_ptr, x_loc_ptr+sz_loc);

        scalar_type res = 0.0;
        // std::cout << "sz_loc = " << sz_loc << " res_loc = "  << res_loc << std::endl;
        hypre_MPI_Allreduce(
            &res_loc, 
            &res, 
            1,
            HYPRE_MPI_REAL, 
            hypre_MPI_SUM, 
            mpi_data_.comm);
        hypre_MPI_Barrier(mpi_data_.comm); 

        return res;
    }
    [[nodiscard]] scalar_type asum(const vector_type &x)const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::asum");

        auto x_loc = hypre_ParVectorLocalVector( x.data() );
        auto sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 
        auto res_loc = HYPRE_THRUST_CALL(reduce, x_loc_ptr, x_loc_ptr+sz_loc, 0.0, sum_opearation<scalar_type>() );

        scalar_type res = 0.0;
        // std::cout << "sz_loc = " << sz_loc << " res_loc = "  << res_loc << std::endl;
        hypre_MPI_Allreduce(
            &res_loc, 
            &res, 
            1,
            HYPRE_MPI_REAL, 
            hypre_MPI_SUM, 
            mpi_data_.comm);
        hypre_MPI_Barrier(mpi_data_.comm); 

        return res;        
    }
    [[nodiscard]] bool is_valid_number(const vector_type &x)const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::is_valid_number");

        if( std::isfinite( norm(x) ) )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::scalar_prod");

        scalar_type res = 0;
        HYPRE_SAFE_CALL( HYPRE_ParVectorInnerProd( x.data(), y.data(), &res) );
        return res;
    }
    [[nodiscard]] scalar_type norm(const vector_type &x) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::norm");

        return std::sqrt( scalar_prod(x,x) );
    }
    [[nodiscard]] scalar_type norm_sq(const vector_type &x) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::norm_sq");

        return scalar_prod(x,x);
    }
    [[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::norm2_sq");

        auto global_size = hypre_ParVectorGlobalSize(x.data());
        return  scalar_prod(x,x)/global_size;
    }
    [[nodiscard]] scalar_type norm2(const vector_type &x) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::norm2");
        
        return std::sqrt( norm2_sq(x) );
    } 
    [[nodiscard]] vector_type at(multivector_type& x, ordinal_type m, ordinal_type k_)
    {
        if (k_ < 0 || k_>=m  ) 
        {
            throw std::out_of_range("hypre_operations: multivector.at");
        }
        // std::cout << "at:" << std::endl;
        return x[k_];
    }

    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::add_lin_comb(mul_x,x,mul_y,y)");

        HYPRE_SAFE_CALL(HYPRE_ParVectorScale(mul_y, y.data()));
        HYPRE_SAFE_CALL( HYPRE_ParVectorAxpy(mul_x, x.data(), y.data() ) );

    }
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::add_lin_comb(mul_x,x,mul_y,y,mul_z,z)");

        add_lin_comb(mul_x, x, mul_z, z);
        add_lin_comb(mul_y, y, 1.0, z);
    }    

    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign_lin_comb");

        assign_scalar(0, y);
        HYPRE_SAFE_CALL( HYPRE_ParVectorAxpy(mul_x, x.data(), y.data() ) );
    }
    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign_lin_comb");

        assign_scalar(0, z);
        add_lin_comb(mul_x, x, mul_y, y, 0, z);
    }
    // y = mul*y.*x in Matlab notation.
    void poinwise_mul(const scalar_type mul, const vector_type& x, vector_type& y)
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::poinwise_mul");
        hypre_Vector *x_local = hypre_ParVectorLocalVector( x.data() );
        hypre_Vector *y_local = hypre_ParVectorLocalVector( y.data() );

        scalar_type *x_data = hypre_VectorData(x_local);
        scalar_type *y_data = hypre_VectorData(y_local);
        ordinal_type size_x = hypre_VectorSize(x_local);
        ordinal_type size_y = hypre_VectorSize(y_local);

        if(size_x != size_y)
        {
            throw std::runtime_error("hypre_operations::poinwise_mul size_x != size_y");
        }

        for_each( detail::kernel::mul_pointwise<ordinal_type, scalar_type, scalar_type*>{mul, x_data, y_data}, size_x );
    }

    // matrix_vector_operations

    //calc: y := alpha*mat*x + beta*y
    void add_matrix_vector_prod(const scalar_type alpha, const matrix_type& mat, const vector_type& x, const scalar_type beta, vector_type& y) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::add_matrix_vector_prod");

        HYPRE_SAFE_CALL( hypre_ParCSRMatrixMatvec(alpha, mat.data(), x.data(), beta, y.data() ) );
    }
    //calc: z := alpha*mat*x + beta*y
    void assign_matrix_vector_prod(const scalar_type alpha, const matrix_type& mat, const vector_type& x, const scalar_type beta, const vector_type& y, vector_type& z) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::assign_matrix_vector_prod");

        assign(y, z);
        add_matrix_vector_prod(alpha, mat, x, beta, z);
    }

    [[nodiscard]] std::shared_ptr<vector_space_type> get_matrix_im_space(const matrix_type& mat) const;
    [[nodiscard]] std::shared_ptr<vector_space_type> get_matrix_dom_space(const matrix_type& mat) const;
    

    // matrix_operations

    // [[nodiscard]] matrix_type matrix_transpose(const matrix_type &mat) const
    // {
    //     HYPRE_ParCSRMatrix parcsr_D;
    //     HYPRE_SAFE_CALL( hypre_ParCSRMatrixTranspose(mat.data(), &parcsr_D, 1 ) ); // data = 1 to get data to new matrix
    //     return matrix_type(mpi_data_, parcsr_D);
    // }


    // [[nodiscard]] matrix_type matrix_matrix_prod(const matrix_type &mat_a, const matrix_type &mat_b)const
    // {
        
    //     HYPRE_ParCSRMatrix parcsr_D = hypre_ParCSRMatMat( mat_a.data(), mat_b.data() );
    //     return matrix_type(mpi_data_, parcsr_D);

    // }
    // [[nodiscard]] matrix_type matrix_matrix_sum(const scalar_type alpha, const matrix_type &mat_a, const scalar_type beta, const matrix_type &mat_b)const
    // {
        
    //     HYPRE_ParCSRMatrix parcsr_D;
    //     HYPRE_SAFE_CALL(hypre_ParCSRMatrixAdd( alpha, mat_a.data(), beta, mat_b.data(), &parcsr_D) );
    //     return matrix_type(mpi_data_, parcsr_D);
    // }
    // [[nodiscard]] scalar_type matrix_norm_fro(const matrix_type &mat) const
    // {
    //     return hypre_ParCSRMatrixFnorm( mat.data() );
    // }
    // [[nodiscard]] matrix_type matrix_diag(const matrix_type &mat, bool invert = false) const
    // {
    //     // Problem. Only available function is:
    //     // void hypre_CSRMatrixExtractDiagonal( hypre_CSRMatrix *A,
    //                             // HYPRE_Complex   *d,
    //                             // HYPRE_Int        type)
    //     // It returns a VECTOR and has the following parametrization:
    //     /*--------------------------------------------------------------------------
    //      * hypre_CSRMatrixExtractDiagonal
    //      *
    //      * type 0: diag
    //      *      1: abs diag
    //      *      2: diag inverse
    //      *      3: diag inverse sqrt
    //      *--------------------------------------------------------------------------*/
    //     HYPRE_ParCSRMatrix parcsr_D;
    //     throw std::logic_error("NOT IMPLEMENTED YET");
    //     return matrix_type(mpi_data_, parcsr_D);
    // }

    [[nodiscard]] std::shared_ptr<matrix_type> matrix_transpose(const matrix_type &mat) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_transpose");

        HYPRE_ParCSRMatrix parcsr_D;
        HYPRE_SAFE_CALL( hypre_ParCSRMatrixTranspose(mat.data(), &parcsr_D, 1 ) ); // data = 1 to get data to new matrix
        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);
    }


    [[nodiscard]] std::shared_ptr<matrix_type> matrix_matrix_prod(const matrix_type &mat_a, const matrix_type &mat_b)const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_matrix_prod");
        
        HYPRE_ParCSRMatrix parcsr_D = hypre_ParCSRMatMat( mat_a.data(), mat_b.data() );
        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);

    }
    [[nodiscard]] std::shared_ptr<matrix_type> matrix_matrix_sum(const scalar_type alpha, const matrix_type &mat_a, const scalar_type beta, const matrix_type &mat_b)const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_matrix_sum");
        
        HYPRE_ParCSRMatrix parcsr_D;
        HYPRE_SAFE_CALL(hypre_ParCSRMatrixAdd( alpha, mat_a.data(), beta, mat_b.data(), &parcsr_D) );
        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);
    }
    [[nodiscard]] scalar_type matrix_norm_fro(const matrix_type &mat) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_norm_fro");

        return hypre_ParCSRMatrixFnorm( mat.data() );
    }
    [[nodiscard]] std::shared_ptr<matrix_type> matrix_diag(const matrix_type &mat, bool invert = false) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_diag");

        HYPRE_ParCSRMatrix parcsr_A = mat.data();
        /// Orgignally was taken from hypre: src/test/ij_mm.c:599 (function runjob5), 
        /// but hypre_CSRMatrixInitialize_v2 was added (see comment below)
        HYPRE_ParCSRMatrix parcsr_D = 
            hypre_ParCSRMatrixCreate(
                hypre_ParCSRMatrixComm(parcsr_A),
                hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                hypre_ParCSRMatrixRowStarts(parcsr_A),
                hypre_ParCSRMatrixColStarts(parcsr_A),
                0,
                hypre_ParCSRMatrixNumRows(parcsr_A),
                0
            );
        hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_D));
        hypre_ParCSRMatrixDiag(parcsr_D) = 
            hypre_CSRMatrixDiagMatrixFromMatrixDevice(
                hypre_ParCSRMatrixDiag(parcsr_A), (invert?2:0)
            );
        /// Original version from sample ij_mm.c worked only for single GPU; this line remedies this problem:
        /// function hypre_ParCSRMatrixCreate creates matrix with matrices with no data (empty ptr,col,var).
        /// Extraction of diagonal matrix above create actual diag matrix. However matrix-multiplication function
        /// in mGPU requieres coherent offdiag part. Dispite offd actyally does not contain any elements its ptr
        /// array should be initialized (as it occured). Unfortunatly, I can not find any evidence that all this is 
        /// enough and correct - try and error only
        /// TODO find more information of that
        hypre_CSRMatrixInitialize_v2( hypre_ParCSRMatrixOffd(parcsr_D), 0, HYPRE_MEMORY_DEVICE );

        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);
    }

    /// Returns diagonal of the matrix as vector (vector must already be allocated and have corresponding partitioning)
    void matrix_diag(const matrix_type &mat, vector_type& x, bool invert = false) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::matrix_diag");

        HYPRE_ParCSRMatrix parcsr_A = mat.data();

        auto x_loc = hypre_ParVectorLocalVector( x.data() );
        auto sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 

        /// hypre_CSRMatrixExtractDiagonal copies diagonal values data to x vector buffer, see its description in hypre
        hypre_CSRMatrixExtractDiagonal( hypre_ParCSRMatrixDiag(parcsr_A), x_loc_ptr, (invert?2:0));

    }
    /// Creates a matrix with diagonal structure and values on its diagonal from vector x
    [[nodiscard]] std::shared_ptr<matrix_type> diag_matrix_from_vector(const vector_type& x) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::diag_matrix_from_vector");

        HYPRE_ParVector parvec_x = x.data();
        auto x_sz_glob = hypre_ParVectorGlobalSize(parvec_x);
        auto x_loc = hypre_ParVectorLocalVector(parvec_x);
        auto x_sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 
        /// the same as in first matrix_diag method but filles with vector x parameters 
        HYPRE_ParCSRMatrix parcsr_D = 
            hypre_ParCSRMatrixCreate(
                hypre_ParVectorComm(parvec_x),
                x_sz_glob,
                x_sz_glob,
                hypre_ParVectorPartitioning(parvec_x),
                hypre_ParVectorPartitioning(parvec_x),
                0,
                x_sz_loc,
                0
            );
        ///This was a first attempt to init and fill matrix manually
        /*
        /// opposed to matrix_diag method we donot recreate diag part of new matrix but just create its data arrays and 
        /// copy values from x there
        auto D_diag = hypre_ParCSRMatrixDiag(parcsr_D);
        /// NOTE hypre_ParCSRMatrixCreate doesnot fill data arrays for internal local matrices
        hypre_CSRMatrixInitialize_v2( D_diag, 0, HYPRE_MEMORY_DEVICE );
        /// TODO do we need safe call here?
        hypre_TMemcpy(
            hypre_CSRMatrixData(D_diag),             ///dst
            x_loc_ptr,                               ///src
            scalar_type,                             ///type to copy
            x_sz_loc,                                ///size to copy in elements of scalar_type count
            HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE ///src and dst locations
        );
        */
        /// This is second attempt that looks like the one from 1st matrix_diag but  
        /// uses function hypre_CSRMatrixDiagMatrixFromVectorDevice (found in hypre_CSRMatrixDiagMatrixFromMatrixDevice implementation)
        hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_D));
        hypre_ParCSRMatrixDiag(parcsr_D) = 
            hypre_CSRMatrixDiagMatrixFromVectorDevice(x_sz_loc, x_loc_ptr);
        /// see comment in the same part of the first matrix_diag method
        hypre_CSRMatrixInitialize_v2( hypre_ParCSRMatrixOffd(parcsr_D), 0, HYPRE_MEMORY_DEVICE );

        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);
    }
    /// Creates a matrix with diagonal structure and scalar value on its diagonal val
    /// Parallel structure defined by vector x
    /// TODO make something better then explicit vector!!
    [[nodiscard]] std::shared_ptr<matrix_type> scalar_matrix(const vector_type& x,scalar_type val) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::scalar_matrix");

        HYPRE_ParVector parvec_x = x.data();
        auto x_sz_glob = hypre_ParVectorGlobalSize(parvec_x);
        auto x_loc = hypre_ParVectorLocalVector(parvec_x);
        auto x_sz_loc = hypre_VectorSize(x_loc);
        auto x_loc_ptr = hypre_VectorData(x_loc); 
        /// the same as in first matrix_diag method but filles with vector x parameters 
        HYPRE_ParCSRMatrix parcsr_D = 
            hypre_ParCSRMatrixCreate(
                hypre_ParVectorComm(parvec_x),
                x_sz_glob,
                x_sz_glob,
                hypre_ParVectorPartitioning(parvec_x),
                hypre_ParVectorPartitioning(parvec_x),
                0,
                x_sz_loc,
                0
            );
        hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_D));
        hypre_ParCSRMatrixDiag(parcsr_D) = 
            hypre_CSRMatrixIdentityDevice(x_sz_loc, val);
        /// see comment in the same part of the first matrix_diag method
        hypre_CSRMatrixInitialize_v2( hypre_ParCSRMatrixOffd(parcsr_D), 0, HYPRE_MEMORY_DEVICE );

        return std::make_shared<matrix_type>(mpi_data_, parcsr_D);
    }


    void write_matrix_to_mm_file(const std::string& file_name, const matrix_type& mat) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::write_matrix_to_mm_file");
        mat.save(file_name);
    }

    void write_matrix_to_mm_file(const std::string& file_name, std::shared_ptr<matrix_type> mat) const
    {
        STOKES_PORUS_3D_PLATFORM_SCOPED_TIC("hypre_operations::write_matrix_to_mm_file");
        mat->save(file_name);
    }


    struct add_mul_scalar_functor
    {
        add_mul_scalar_functor(scalar_type scal, scalar_type mul):
        scal_(scal), mul_(mul)
        {}

        __DEVICE_TAG__ scalar_type operator()(scalar_type val) const 
        { 
            
            return (val*mul_+scal_);
        }
    private:
        scalar_type scal_;
        scalar_type mul_;      
    };
    template<class TT>
    struct sum_opearation
    {
        __DEVICE_TAG__ TT operator()(const TT &x1, const TT &x2) const
        {
            TT y1 = x1 < static_cast<scalar_type>(0) ? -x1 : x1;
            TT y2 = x2 < static_cast<scalar_type>(0) ? -x2 : x2;
            return y1+y2;
        }
    };

   

};


}   // namespace linspace
}   // namespace scfd

#endif