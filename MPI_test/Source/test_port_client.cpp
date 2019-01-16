#include <mpi.h>
#include <iostream>
#include <cstring>
#include <stdio.h>
#define MAX_DATA 10


int main( int argc, char **argv ) 
{ 
    MPI_Comm server; 
    double buf[MAX_DATA]; 
    char port_name[MPI_MAX_PORT_NAME]; 
    int size, rank;

    MPI_Init( &argc, &argv ); 
    printf("\n1\n");
    std::strcpy(port_name, argv[1]);/* assume server's name is cmd-line arg */ 
    printf("\n%s\n", port_name);

    

    MPI_Comm_connect( port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD,  
                      &server ); 
    printf("\n4\n");
    int tag=2;
    while (tag>1) 
    { 
        std::cout << "tag>>>";
        std::cin >> tag; 
        for (int i = 0; i < MAX_DATA; ++i)
        {
            buf[i]=buf[i]+tag;
        }
        MPI_Send( buf, MAX_DATA, MPI_DOUBLE, 0, tag, server ); 
        
    } 
    MPI_Comm_disconnect( &server ); 
    MPI_Finalize(); 
    return 0; 
} 