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

#ifndef __CUSPARSE_SAFE_CALL_H__
#define __CUSPARSE_SAFE_CALL_H__

#include <stdexcept>
#include <string>
#include <sstream>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define CUSPARSE_SAFE_CALL(X)                                                                                                                                                                                                                          \
        do {                                                                                                                                                                                                                                           \
                cusparseStatus_t status = (X);                                                                                                                                                                                                         \
                cudaError_t cuda_res = cudaDeviceSynchronize();                                                                                                                                                                                        \
                if (status != CUSPARSE_STATUS_SUCCESS) {                                                                                                                                                                                               \
                        std::stringstream ss;                                                                                                                                                                                                          \
                        ss << std::string("CUSPARSE_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: returned status ") << status;                                                                                                          \
                        std::string str = ss.str();                                                                                                                                                                                                    \
                        throw std::runtime_error(str);                                                                                                                                                                                                 \
                }                                                                                                                                                                                                                                      \
                if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUSOLVER_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed cudaDeviceSynchronize: ") + std::string(cudaGetErrorString(cuda_res)));                         \
        } while (0)

#endif
