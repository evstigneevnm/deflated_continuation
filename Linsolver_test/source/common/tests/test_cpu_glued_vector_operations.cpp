#include <iostream>
#include <string>
#include <utils/log.h>
#include <common/cpu_vector_operations.h>
#include <common/glued_vector_operations.h>
#include <common/glued_vector_space.h>




int main(int argc, char const *argv[]) 
{
    using log_t = utils::log_std;

    using real = SCALAR_TYPE;
    using T = real;
    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using glued_vec_space_t = scfd::linspace::glued_vector_space<vec_ops_t, 2>;
    using glued_vec_t = typename glued_vec_space_t::vector_type;



    T ref_error = std::numeric_limits<T>::epsilon();

    if(argc != 2)
    {
        std::cout << "usage: " << argv[0] << " size" << std::endl;
        return 2;
    }

    std::size_t N_loc = std::stoi(argv[1]);

    log_t log;
    log.info("test glued vector operations");

    std::shared_ptr<vec_ops_t> vec_ops1 = std::make_shared<vec_ops_t>(N_loc);
    std::shared_ptr<vec_ops_t> vec_ops2 = std::make_shared<vec_ops_t>(2*N_loc);
    std::shared_ptr<glued_vec_space_t> glued_vec_space = std::make_shared<glued_vec_space_t>(vec_ops1,vec_ops2);
    log.info_f("vec1 size = %i", vec_ops1->size() ); 
    log.info_f("vec2 size = %i", vec_ops2->size() ); 

    std::size_t N = vec_ops1->size() + vec_ops2->size();


    glued_vec_t x,y;
    glued_vec_space->init_vector(x);
    glued_vec_space->init_vector(y);
    glued_vec_space->start_use_vector(x);
    glued_vec_space->start_use_vector(y);
    
    glued_vec_space->assign_scalar(2.0, x);
    glued_vec_space->assign_scalar(1.0, y);
    auto dot_x_y = glued_vec_space->scalar_prod(x,y);
    auto err = std::abs(dot_x_y-N*(2*1));
    log.info_f("(x,y) = %e, error = %e", dot_x_y, err );
    err = err + std::abs(glued_vec_space->norm(x)-std::sqrt(N*2*2));
    log.info_f("norm x = %e, error = %e", glued_vec_space->norm(x), err );
    err = err + std::abs(glued_vec_space->norm(y)-std::sqrt(N*1*1));
    log.info_f("norm y = %e, error = %e", glued_vec_space->norm(y), err );
    glued_vec_space->add_mul(2.0, x, 2.0, y);
    err = err + std::abs(glued_vec_space->norm(y)-std::sqrt(N*6*6));
    log.info_f("add_mul(2,x,2,y) norm y = %e, error = %e", glued_vec_space->norm(y), err );
    
    
    glued_vec_space->stop_use_vector(x);
    glued_vec_space->free_vector(x);
    glued_vec_space->stop_use_vector(y);
    glued_vec_space->free_vector(y);

    vec_t xv, yv;
    vec_ops1->init_vector(xv); vec_ops1->start_use_vector(xv);
    vec_ops2->init_vector(yv); vec_ops2->start_use_vector(yv);

    glued_vec_t xy;
    glued_vec_space->init_vector(xy); glued_vec_space->start_use_vector(xy);

    vec_ops1->assign_scalar(2.0,xv);
    vec_ops2->assign_scalar(3.0,yv);
    vec_ops1->assign(xv, xy.comp(0));
    vec_ops2->assign(yv, xy.comp(1));
    auto xy_norm = glued_vec_space->norm(xy);
    err = std::abs(xy_norm - std::sqrt( N_loc*(2*2)+2*N_loc*(3*3) ) );
    log.info_f("norm of two induced vectors = %e, error = %e", xy_norm, err );    

    vec_ops1->stop_use_vector(xv); vec_ops1->free_vector(xv);
    vec_ops2->stop_use_vector(yv); vec_ops2->free_vector(yv);
    glued_vec_space->stop_use_vector(xy); glued_vec_space->free_vector(xy);


    int N_tests = 6;
    auto ref_error_val = ref_error*N_tests*std::sqrt(N);
    if(err > ref_error*N_tests*std::sqrt(N) )
    {
        log.error_f("Got error = %e with reference = %e", err,  ref_error_val);
        return 1;
    }
    else
    {
        log.info_f("No errors with reference = %e", ref_error_val);
        return 0;    
    }

}