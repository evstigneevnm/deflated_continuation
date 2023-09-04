import numpy as np
import sys


def main():
    argc = len(sys.argv)
    if argc != 2:
        print("usage: ", sys.argv[0], " system_size");
        return 0
    N = int(sys.argv[1])
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    b = np.random.rand(N)

    C_geam = A+B
    C_gemm = np.matmul(A,B)
    C1_gemm = np.matmul(A,B) + C_gemm;
    CTN_gemm = np.matmul( np.transpose(A), B )
    CNT_gemm = np.matmul( A, np.transpose(B) )
    x = np.linalg.solve(A, b);
    iA = np.linalg.inv(A)

    np.savetxt("A.dat", A)
    np.savetxt("B.dat", B)
    np.savetxt("C_geam.dat", C_geam)
    np.savetxt("C_gemm.dat", C_gemm)
    np.savetxt("C1_gemm.dat", C1_gemm)
    np.savetxt("CTN_gemm.dat", CTN_gemm)
    np.savetxt("CNT_gemm.dat", CNT_gemm)
    np.savetxt("b.dat", b)
    np.savetxt("x.dat", x)
    np.savetxt("iA.dat", iA)

if __name__ == '__main__':
    main()