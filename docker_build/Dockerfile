FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 AS build
RUN apt-get update
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install openmpi-bin openmpi-common
RUN apt-get -y install vim tmux
WORKDIR /stokes
ADD ./Stokes_porus_3D_cuda/ /stokes
ADD ./ccl /ccl
RUN cd /stokes/sourse/contrib/hypre/src/ && make clean && ./configure --with-MPI --enable-gpu-aware-mpi --with-MPI-include=/usr/lib/x86_64-linux-gnu/openmpi/include --with-MPI-lib-dirs=/usr/lib/x86_64-linux-gnu/openmpi/lib --enable-cuda-streams --with-cuda-home=/usr/local/cuda-11.6/ --with-gpu-arch='60 61 62 70 72 75 80 86' --enable-unified-memory --enable-single && make -j 48
RUN cd /stokes && make stokes_hypre_single.bin CONFIG_FILE=config_docker_release.inc && make stokes_hypre_omp_single.bin CONFIG_FILE=config_docker_release.inc
RUN cd /stokes/sourse/contrib/hypre/src/ && make clean && ./configure --with-MPI --enable-gpu-aware-mpi --with-MPI-include=/usr/lib/x86_64-linux-gnu/openmpi/include --with-MPI-lib-dirs=/usr/lib/x86_64-linux-gnu/openmpi/lib --enable-cuda-streams --with-cuda-home=/usr/local/cuda-11.6/ --with-gpu-arch='60 61 62 70 72 75 80 86' --enable-unified-memory && make -j 48
RUN cd /stokes && make stokes_hypre_double.bin CONFIG_FILE=config_docker_release.inc && make stokes_hypre_omp_double.bin CONFIG_FILE=config_docker_release.inc




FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04
RUN apt-get update
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install openmpi-bin openmpi-common
WORKDIR /stokes
COPY --from=build /stokes/stokes_hypre*.bin /stokes/



