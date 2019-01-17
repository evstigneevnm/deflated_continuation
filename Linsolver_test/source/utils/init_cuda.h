// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __INITCUDA_H__
#define __INITCUDA_H__

#include <stdio.h>
#include <cuda_runtime.h>

namespace utils
{

#if __DEVICE_EMULATION__

inline bool init_cuda(int user_prefered_i = -1){ return true; }

#else
inline bool init_cuda(int user_prefered_i = -1)
{
    printf("Trying to init cuda\n");
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    if (user_prefered_i == -1) {
        for(i = 0; i < count; i++) {
            cudaDeviceProp prop;
            if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                if(prop.major >= 1) {
                    break;
                }
            }
        }
        if(i == count) {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return false;
        }
    } else i = user_prefered_i;
    if (cudaSetDevice(i) != cudaSuccess) {
        fprintf(stderr, "Error cudaSetDevice.\n");
        return false;
    }

    printf("CUDA initialized; device number %d.\n", i);
    return true;
}

#endif

}

#endif