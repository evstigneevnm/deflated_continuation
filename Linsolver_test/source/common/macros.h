#ifndef __ARNOLDI_MACROS_H__
#define __ARNOLDI_MACROS_H__

#ifndef I2_R
	#define I2_R(i , j, Rows) (i)+(j)*(Rows)
#endif

#ifndef I2
    #define I2(i, j, Col) (i)*(Col)+(j)
#endif

#ifndef I2P
    #define I2P(j, k) ((j)>(Nx-1)?(j)-Nx:(j)<0?(Nx+(j)):(j))*(Ny)+((k)>(Ny-1)?(k)-Ny:(k)<0?(Ny+(k)):(k))
#endif

#ifndef _I3
    #define _I3(i, j, k, Nx, Ny, Nz) (i)*(Ny*Nz) + (j)*(Nz) + (k)
#endif

#ifndef I3 //default for Nx, Ny, Nz
    #define I3(i, j, k) _I3(i, j, k, Nx, Ny, Nz)
#endif

#ifndef I3P
	#define I3P(i, j, k) (Ny*Nz)*((i)>(Nx-1)?(i)-Nx:(i)<0?(Nx+(i)):(i))+((j)>(Ny-1)?(j)-Ny:(j)<0?(Ny+(j)):(j))*(Nz)+((k)>(Nz-1)?(k)-Nz:(k)<0?(Nz+(k)):(k))
#endif

#endif