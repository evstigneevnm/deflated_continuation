#include <mpi.h>
#include <stdio.h>

#define MAX_DATA 10

int main(int argc, char *argv[]) 
{ 
    MPI_Comm client; 
    MPI_Status status; 
    char port_name[MPI_MAX_PORT_NAME]; 
    double buf[MAX_DATA]; 
    int    size, again, rank; 
 
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size != 1)
    {
        printf("nproc > 1. Working only on proc=0\n");
        MPI_Finalize(); 
        return 0;
    }
    
 //   if(rank==0)
 //   {
        MPI_Open_port(MPI_INFO_NULL, port_name); 
        printf("server available at %s\n", port_name); 
    
        while (1) 
        { 
            MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD,  
                             &client); 
            again = 1; 
            while(again==1) 
            { 
                MPI_Recv(buf, MAX_DATA, MPI_DOUBLE,  
                          MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status); 
                switch (status.MPI_TAG) 
                { 
                    case 0: printf("Server got close tag\n");
                            MPI_Comm_free(&client); 
                            MPI_Close_port(port_name); 
                            MPI_Finalize(); 
                            return 0; 
                    case 1: printf("Server got disconnect tag\n");
                            MPI_Comm_disconnect(&client); 
                            again = 0; 
                            break; 
                    case 2:
                            printf("Server got message tag 2\n");
                            for (int j = 0; j < MAX_DATA; ++j)
                            {
                                printf("%lf",buf[j]);
                            }
                            printf("\n");
                            break;
                    default: 
                            /* Unexpected message type */ 
                            printf("Server got unexpected message tag\n");
                            MPI_Comm_disconnect(&client); 
                            again = 0; 
                            break;
                } 
            } 
        }
 //  }
    MPI_Finalize(); 
    return 0;

} 