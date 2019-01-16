#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double ans;
    if(rank == 0){
        fread(&ans, sizeof(ans), 1, stdin);
    }

    MPI_Bcast(&ans, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("rank %d of %d received %lf\n", rank, size, ans);
    MPI_Finalize();
}
