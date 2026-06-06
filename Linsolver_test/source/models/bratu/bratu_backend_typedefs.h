#ifndef __BRATU_BACKEND_TYPEDEFS_H__
#define __BRATU_BACKEND_TYPEDEFS_H__

#ifndef Blocks_x_
#define Blocks_x_ 64
#endif

#if defined(BRATU_VECTOR_BACKEND_VAR_PREC)
#include <common/cpu_vector_operations_var_prec.h>
#elif defined(BRATU_VECTOR_BACKEND_OMP)
#include <common/scfd_vector_operations.h>
#include <scfd/backend/omp.h>
#else
#error "Bratu first pass supports BRATU_VECTOR_BACKEND_OMP and BRATU_VECTOR_BACKEND_VAR_PREC targets."
#endif

#include <scfd/utils/log.h>

#if defined(BRATU_VECTOR_BACKEND_VAR_PREC)
#ifndef BRATU_VAR_PREC_BITS
#define BRATU_VAR_PREC_BITS 100
#endif
#define BRATU_BACKEND_NAME "cpu_var_prec"
typedef cpu_vector_operations_var_prec<BRATU_VAR_PREC_BITS> vec_ops_real;
typedef typename vec_ops_real::scalar_type real;
#elif defined(BRATU_VECTOR_BACKEND_OMP)
#define BRATU_BACKEND_NAME "scfd_omp"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::omp, real> vec_ops_real;
#endif

typedef scfd::utils::log_std log_t;

#endif
