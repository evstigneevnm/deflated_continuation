#ifndef __SCFD_LINSPACE_HYPRE_VECTOR_SPACE_H__
#define __SCFD_LINSPACE_HYPRE_VECTOR_SPACE_H__

#include <common/hypre_operations.h>


namespace scfd
{
namespace linspace
{

class hypre_vector_space : public hypre_operations
{
    using parent_t = hypre_operations;
public:
    using typename parent_t::ordinal_type;
    using typename parent_t::big_ordinal_type;
    using typename parent_t::vector_type;
    using typename parent_t::multivector_type;
    using typename parent_t::scalar_type;
    using typename parent_t::matrix_type;
    using typename parent_t::comm_info_type;
    using Ord = parent_t::ordinal_type;
    using BigOrd = parent_t::big_ordinal_type;
private:

    big_ordinal_type part_[2];
    big_ordinal_type global_sz_;

public:

    template<class VecPart>
    hypre_vector_space(const comm_info_type& mpi_p, const VecPart& part_p):
    parent_t(mpi_p), part_{part_p[0], part_p[1]}
    {
        big_ordinal_type local_segment_end = static_cast<big_ordinal_type>(part_p[1]);

        hypre_MPI_Allreduce(
            &local_segment_end, 
            &global_sz_, 
            1, 
            HYPRE_MPI_BIG_INT, 
            hypre_MPI_MAX, 
            parent_t::mpi_data_.comm);

        hypre_MPI_Barrier(parent_t::mpi_data_.comm);
        // std::cout << "global_sz_ = " << global_sz_ << std::endl;
    }

    ~hypre_vector_space()
    {}

    big_ordinal_type size()const 
    {
        return global_sz_;
    }

    void init_vector(vector_type& x) const
    {
        vector_type xx(parent_t::mpi_data_, part_);
        x = std::move(xx);
    }
    void start_use_vector(vector_type& x)const
    {}
    void stop_use_vector(vector_type& x)const
    {}
    void free_vector(vector_type& x)const
    {}

    void init_multivector(multivector_type& x, ordinal_type m) const
    {
        x = multivector_type();
        x.reserve(m);
        for(ordinal_type j=0;j<m;j++)
        {
            x.emplace_back(parent_t::mpi_data_, part_);
        }
    }
    void free_multivector(multivector_type& x, ordinal_type m) const
    {
    }
    void start_use_multivector(multivector_type& x, ordinal_type m) const
    {
    }
    void stop_use_multivector(multivector_type& x, ordinal_type m) const
    {
    }

}; 



[[nodiscard]] std::shared_ptr<hypre_vector_space> hypre_operations::get_matrix_im_space(const matrix_type& mat) const
{
    return std::make_shared<hypre_vector_space>(hypre_operations::mpi_data_, mat.row_partition() ); //amgcl::backend::rows(*mat), *this
}
[[nodiscard]] std::shared_ptr<hypre_vector_space> hypre_operations::get_matrix_dom_space(const matrix_type& mat) const
{
    return std::make_shared<hypre_vector_space>(hypre_operations::mpi_data_, mat.col_partition() );
}



}
}


#endif