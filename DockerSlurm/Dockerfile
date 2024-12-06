FROM nvcr.io/nvidia/hpc-benchmarks:24.06 as build
RUN apt-get update
RUN apt-get upgrade -y
RUN apt install libboost-all-dev -y
#RUN apt-get install gfortran -y
#building solver
WORKDIR /continuation
ADD ./Linsolver_test/source /continuation/source
ADD ./Linsolver_test/O /continuation/O
ADD ./Linsolver_test/Makefile /continuation/Makefile
ADD ./Linsolver_test/config_dockerslurm_rel.inc /continuation/config_dockerslurm_rel.inc
ADD ./Linsolver_test/json_project_files/config_file_continuation.json /continuation/config_file_continuation.json
ADD ./240806/ /continuation/start_data
RUN rm *.bin || true
RUN rm ./O/*.o || true
RUN cd /continuation
RUN make ker CONFIG_FILE=config_dockerslurm_rel.inc -B || cat results.make
RUN make Kolmogorov_3D_all CONFIG_FILE=config_dockerslurm_rel.inc -B || cat results.make
RUN make test_Kolmogorov_3D_continuation_file CONFIG_FILE=config_dockerslurm_rel.inc -B || cat results.make


FROM nvcr.io/nvidia/hpc-benchmarks:24.06
#copy solver:
WORKDIR /continuation
COPY --from=build /continuation/test_Kolmogorov_3D_continuation_file.bin /continuation/.
COPY --from=build /continuation/config_file_continuation.json /continuation/.
COPY --from=build /continuation/start_data /continuation/.

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
