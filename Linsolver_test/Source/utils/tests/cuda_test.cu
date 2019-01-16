
#include <stdio.h>
#include <string>
#include <utils/InitCUDA.h>
#include <utils/cuda_safe_call.h>

int main(int argc, char **argv)
{
        bool    do_error = false;
        if ((argc >= 2)&&(std::string(argv[1]) == std::string("1"))) do_error = true;
        if (do_error) printf("you specified do error on purpose\n");
        try {
                if (!InitCUDA(0)) throw std::runtime_error("InitCUDA failed");

                int     *p;
                if (!do_error)
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, sizeof(int)*512) );
                else
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, -100 ) );

                return 0;

        } catch (std::runtime_error &e) {
                printf("%s\n", e.what());

                return 1;
        }
}