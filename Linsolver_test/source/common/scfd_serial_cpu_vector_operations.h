#ifndef __COMMON_SCFD_SERIAL_CPU_VECTOR_OPERATIONS_H__
#define __COMMON_SCFD_SERIAL_CPU_VECTOR_OPERATIONS_H__

#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>

template<class T>
using scfd_serial_cpu_vector_operations = scfd_vector_operations<scfd::backend::serial_cpu, T>;

#endif
