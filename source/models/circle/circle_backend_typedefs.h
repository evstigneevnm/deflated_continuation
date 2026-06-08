#ifndef __CIRCLE_BACKEND_TYPEDEFS_H__
#define __CIRCLE_BACKEND_TYPEDEFS_H__

#ifndef Blocks_x_
#define Blocks_x_ 64
#endif

#if defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
#include <common/cpu_vector_operations_var_prec.h>
#elif defined(CIRCLE_VECTOR_BACKEND_OMP)
#include <common/scfd_vector_operations.h>
#include <scfd/backend/omp.h>
#elif defined(CIRCLE_VECTOR_BACKEND_HIP)
#include <common/scfd_vector_operations.h>
#include <scfd/backend/hip.h>
#else
#include <common/scfd_vector_operations.h>
#include <scfd/backend/cuda.h>
#endif

#include <scfd/utils/log.h>

#if defined(CIRCLE_VECTOR_BACKEND_VAR_PREC)
#ifndef CIRCLE_VAR_PREC_BITS
#define CIRCLE_VAR_PREC_BITS 100
#endif
#define CIRCLE_BACKEND_NAME "cpu_var_prec"
typedef cpu_vector_operations_var_prec<CIRCLE_VAR_PREC_BITS> vec_ops_real;
typedef typename vec_ops_real::scalar_type real;
#elif defined(CIRCLE_VECTOR_BACKEND_OMP)
#define CIRCLE_BACKEND_NAME "scfd_omp"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::omp, real> vec_ops_real;
#elif defined(CIRCLE_VECTOR_BACKEND_HIP)
#define CIRCLE_BACKEND_NAME "scfd_hip"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::hip, real> vec_ops_real;
#else
#define CIRCLE_BACKEND_NAME "scfd_cuda"
typedef SCALAR_TYPE real;
typedef scfd_vector_operations<scfd::backend::cuda, real> vec_ops_real;
#endif

typedef scfd::utils::log_std log_t;

#endif
