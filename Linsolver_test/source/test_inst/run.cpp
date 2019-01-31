#include "class_file.h"

int main(int argc, char const *argv[])
{
    class_file<double> CL(100);
    double *x, *y;
    CL.start_use_vector(x);
    CL.start_use_vector(y);
    CL.add_vectors( (const double*&)x, y );
    CL.stop_use_vector(x);
    CL.stop_use_vector(y);

    return 0;
}