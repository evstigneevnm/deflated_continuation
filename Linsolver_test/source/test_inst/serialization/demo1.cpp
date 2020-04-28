#include <fstream>
#include <iostream>
#include <string>
// include headers that implement a archive
// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
#include <external_libraries/cublas_wrap.h>

#include <utils/cuda_support.h>

#include "test_containers.hpp"

int main(int argc, char const *argv[])
{

    if(argc!=3)
    {
        printf("Usage: %s path_to_project N\n",argv[0]);
        return 0;
    }

    std::string project_dir(argv[1]);

    int Nx=atoi(argv[2]);
    printf("using: %s %s %i\n",argv[0], project_dir.c_str(), Nx);

    std::string test = {};

    typedef double T;
    typedef gpu_vector_operations<T> vec_ops_t;
    typedef typename vec_ops_t::vector_type T_vec;
    typedef gpu_file_operations<vec_ops_t> vec_files_t;
    typedef curve_container<vec_ops_t, vec_files_t> curve_container_t;
    typedef test_containers<vec_ops_t, vec_files_t, curve_container_t> container_t;

    typedef typename boost::archive::binary_oarchive output;
    typedef typename boost::archive::binary_iarchive input;

    init_cuda(1);
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_t vec_ops(Nx, CUBLAS);
    vec_files_t vec_files( (vec_ops_t*) &vec_ops);
    vec_ops_t* vec_ref = &vec_ops;
    vec_files_t* file_ref = &vec_files;
    container_t container(vec_ref, file_ref, project_dir);


    T_vec x;
    vec_ops.init_vector(x); vec_ops.start_use_vector(x);
    
    for(int j=0;j<5;j++)
    {
        container.init_new_curve();
        for(int k=0;k<10;k++)
        {
            vec_ops.assign_random(x);
            container.add(T(k+(j+1)*0.5), x);
        }
        container.close_curve();
    }

    container.print_all();


    
    std::string seri_file_name(project_dir + std::string("/") + std::string("bifurcaiton_diagram.dat"));
    
    std::ofstream save_file(seri_file_name.c_str());
    std::ifstream load_file(seri_file_name.c_str());
    
    std::cout << "saving using serialization..." << std::endl;

    {
        output oa(save_file);
        oa << container;
    }

    container_t *container1 = new container_t(vec_ref, file_ref, project_dir);
    std::cout << "restoring using serialization..." << std::endl;
    {
        input ia(load_file);
        ia >> *container1;
    }

    container1->init_new_curve();
    for(int k=0;k<10;k++)
    {
        vec_ops.assign_random(x);
        container1->add(T(k*1.98238523094321436), x);
    }
    container1->close_curve();
    container1->print_all();
    delete container1;

    vec_ops.stop_use_vector(x); vec_ops.free_vector(x);
    return 0;
}