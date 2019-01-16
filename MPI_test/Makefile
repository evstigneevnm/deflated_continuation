MPI_INCLUDE_DIR = /usr/local/openmpi40/include
MPI_LIB_DIR =/usr/local/openmpi40/lib

deb:
	2>results_make.txt
	gcc -g -I$(MPI_INCLUDE_DIR) Source/test1.cpp  -o O/child.o -c 2>>results_make.txt
	cd O; \
	gcc child.o -L$(MPI_LIB_DIR) -lmpi -lm -o ../child.bin 

comm_server:
	2>results_make.txt
	gcc -g -I$(MPI_INCLUDE_DIR) Source/test_port_server.cpp  -o O/server.o -c 2>>results_make.txt
	cd O; \
	gcc server.o -L$(MPI_LIB_DIR) -lmpi -lm -o ../server.bin 
comm_client:
	2>results_make.txt
	g++ -g -I$(MPI_INCLUDE_DIR) Source/test_port_client.cpp  -o O/client.o -c 2>>results_make.txt
	cd O; \
	g++ client.o -L$(MPI_LIB_DIR) -lmpi -lm -o ../client.bin 