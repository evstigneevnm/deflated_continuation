#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <external_libraries/lapack_wrap.h>


int inline I2_R(int i, int j, int Rows)
{
    return (i)+(j)*(Rows);
}

int read_matrix_size(const std::string &f_name)
{

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix_size: error while opening file " + f_name);
    std::string line;
    int matrix_size=0;
    while (std::getline(f, line)){
        matrix_size++;
    }
    f.close();
    return matrix_size;
}

template <class T>
void print_matrix(size_t Row, size_t Col,  const T* A)
{

    for(int j =0;j<Row;j++)
    {
        for(int k=0;k<Col;k++)
        {
            std::cout << A[I2_R(j, k, Row)] << " ";
        }
        std::cout << std::endl;
    }
} 

template <class T>
void read_matrix(const std::string &f_name,  size_t Row, size_t Col,  T *matrix){
    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix: error while opening file " + f_name);
    for (size_t i = 0; i<Row; i++)
    {
        for(size_t j=0;j<Col;j++)
        {
            // double val=0;  
            // fscanf(stream, "%le",&val);                
            // matrix[I2(i,j,Row)]=(real)val;
            T val;
            f >> val;
            matrix[I2_R(i,j,Row)]=(T)val;
        }
        
    } 

    f.close();
}



int main(int argc, char const *argv[])
{
    



    if(argc == 1)
    {
        lapack_wrap<double> blas(4);
        std::vector<double> A = {
            0.840375529753905, 0.303520794649354,1.71188778298155, 1.35459432800464, 
            -0.888032082329010 , -0.600326562133734, -0.194123535758265, -1.07215528838425,
            0.100092833139322,0.489965321173948,-2.13835526943994,0.960953869740567,
            -0.544528929990548,0.739363123604474,-0.839588747336614,0.124049800003193
        };

        std::vector<double> H = 
        {
            0.840375529753905,-2.20399874041898,0,0,
            0.379221770756618,-1.19316582238836,-2.11045288679646,0,
            -0.809621296570528,0.0972001077911066,-1.06398317901752,-0.462494886927049,
            0.543907634448883,-0.770503013544636,0.772059121073942,-0.357483030164600
        };
        auto H1 = H;

        std::cout << "A:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << A[j+4*k] << " ";
            }
            std::cout << std::endl;
        }

        std::vector<double> eig_r = {0,0,0,0};
        std::vector<double> eig_i = {0,0,0,0};
        std::vector<double> Q = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> R = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        blas.qr(A.data(), 4, Q.data(), R.data() );
        std::cout << "QR(A)" << std::endl << "Q:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << Q[j+4*k] << " ";
            }
            std::cout << std::endl;
        }    
        std::cout << "R:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << R[j+4*k] << " ";
            }
            std::cout << std::endl;
        } 
        blas.qr(A.data(), 4, R.data() );
        std::cout << "Q-diff (should be zero):" << std::endl;    
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << R[j+4*k] - Q[j+4*k] << " ";
            }
            std::cout << std::endl;
        }     

        blas.eigs(A.data(), 4, eig_r.data(), eig_i.data() );
        std::cout << "eigs(A):" << std::endl;
        for(int j=0;j<4;j++)
        {
            std::cout << eig_r[j] << " " << eig_i[j] << std::endl;
        }
        std::vector<std::complex<double>> eig = {0,0,0,0};
        blas.eigs(A.data(), 4, eig.data() );
        std::cout << "eigs(A) complex:" << std::endl;
        for(int j=0;j<4;j++)
        {
            std::cout << eig[j] << std::endl;
        }    

        std::vector<double> eigV = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        // eigV.reserve(4*4);
        blas.eigsv(A.data(), 4, eig_r.data(), eig_i.data(), eigV.data() );
        std::cout << "eigV(A):" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << eigV[j+4*k] << " ";
            }
            std::cout << std::endl;
        }  
        std::vector<std::complex<double>> eigVC = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};    
        blas.eigsv(A.data(), 4, eig.data(), eigVC.data() );    
        std::cout << "eigV(A) complex:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << eigVC[j+4*k] << " ";
            }
            std::cout << std::endl;
        }  

        std::cout << "H:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << H[j+4*k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "eigs(H):" << std::endl;
        blas.hessinberg_eigs(H.data(), 4, eig_r.data(), eig_i.data() );
        for(int j=0;j<4;j++)
        {
            std::cout << eig_r[j] << " " << eig_i[j] << std::endl;
        }
        auto U = H;
        auto T = U;
        
        std::vector<std::complex<double>> eigsH = {0,0,0,0};
        blas.hessinberg_schur(H.data(), 4, U.data(), T.data(), eigsH.data() );
        std::cout << "schur(H): T,U: U*T*U' = H" << std::endl;
        std::cout << "T:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << T[j+4*k] << " ";
            }
            std::cout << std::endl;
        }    
        std::cout << "U:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << U[j+4*k] << " ";
            }
            std::cout << std::endl;
        }   
        std::cout << "col 2 of U:" << std::endl;
        std::vector<double> vec_mat = {0,0,0,0};
        blas.return_col(2, U.data(), 4, vec_mat.data() );
        for(int j=0;j<4;j++)
        {
            std::cout << vec_mat[j] << std::endl;
        }
        std::cout << std::endl;
        std::cout << "row 1 of U:" << std::endl;
        blas.return_row(1, U.data(), 4, vec_mat.data() );
        for(int j=0;j<4;j++)
        {
            std::cout << vec_mat[j] << " ";
        }
        std::cout << std::endl;    
        std::cout << "eigs(H):" << std::endl;
        for(int j=0;j<4;j++)
        {
            std::cout << eigsH[j] << std::endl;
        }    
        std::cout << std::endl;
        auto C = A;
        blas.gemm(A.data(), 'N', H.data(), 'N', 4, C.data() );
        std::cout << "A*H = C:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << C[j+4*k] << " ";
            }
            std::cout << std::endl;
        }      
        blas.gemm(H.data(), 'N', A.data(), 'N', 4, C.data() );
        std::cout << "H*A = C:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << C[j+4*k] << " ";
            }
            std::cout << std::endl;
        }

        // D*F = C \in R^{3X3}
        std::vector<double> D = {1, 2, 1, 2, 0, 4}; //3X2
        std::vector<double> F = {1, -3, 2, 0.5, 4, 3}; //2X3
        blas.gemm('N', 'N', 3, 3, 2, 1.0, D.data(), F.data(), 0, C.data() );

        for(int j=0;j<3;j++)
        {
            for(int k=0;k<3;k++)
            { 
                std::cout << C[j+3*k] << " ";
            }
            std::cout << std::endl;
        }

        auto A2 = A;
        blas.mat_sq( A.data(), 4, A2.data() );
        std::cout << "A^2:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << A2[j+4*k] << " ";
            }
            std::cout << std::endl;
        }


    }
    else if(argc == 2)
    {
        std::string f_name(argv[1]);
        int N = read_matrix_size(f_name);
        lapack_wrap<double> blas(N);
        std::vector<double> A(N*N, 0);
        std::vector<double> A_c(N*N, 0);        
        std::vector<double> Q(N*N, 0);
        std::vector<double> R(N*N, 0);
        std::vector<double> Q_0(N*N, 0);
        read_matrix(f_name,  N, N,  A.data());
        print_matrix(N, N,  A.data() );

        blas.qr(A.data(), N, Q.data(), R.data() );      
        std::cout << "Q:" << std::endl;
        print_matrix(N, N,  Q.data() );
        std::cout << "R:" << std::endl;
        print_matrix(N, N,  R.data() );
        blas.gemm(Q.data(), 'N', R.data(), 'N', N, A_c.data());
        for(int j=0;j<N*N;j++)
        {
            A_c[j] -= A[j];
        }
        std::cout << "A - QR:" << std::endl;        
        print_matrix(N, N,  A_c.data() );        

        std::cout << "submatrix 9X9:" << std::endl;        
        print_matrix(9, 9,  A.data() );                

    }   
    else
    {
        std::cout << "Usage: " << std::endl;
        std::cout << "1:" << argv[0] << std::endl;
        std::cout << "2:" << argv[0] << "matrix_file_name.dat" <<std::endl;
    }


    return 0;
}
