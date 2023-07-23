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
    
    //lapack 


    if(argc == 1)
    {
        using namespace std::complex_literals;

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

        std::vector< std::complex<double> > B = 
        {
            0.0297702368308634 + 0.508894579290508i, 0.154091033772534 + 0.379150875535018i, 0.989940799059916 + 0.254304842122024i, 0.863771403304308 + 0.596021891944303i,
            0.481277122200334 + 0.0930402625569669i, 0.561167878148708 + 0.995592367220847i, 0.836013835411808 + 0.0484916324220905i, 0.761924626541842 + 0.214739790766330i,
            0.676905434158697 + 0.437859128847775i, 0.949325495276995 + 0.869492229395656i, 0.654080037125545 + 0.831838695092381i, 0.165267646441439 + 0.535169282982425i,
            0.402482970825134 + 0.311971481102577i, 0.959295423619886 + 0.947906559807061i, 0.869172938937459 + 0.688725578355128i, 0.0794741568677688 + 0.0557555570829010i,
        };

        std::vector< std::complex<double> > B1(4*4);
        std::vector< std::complex<double> > AC(4*4);

        std::vector<std::complex<double> > A_X_B_ref1 = 
        {
            -0.483082141821235 - 0.208132660800410i, -0.425092571135247 - 0.918007566698950i, -0.298701024217480 - 0.612324962995601i, -0.469926268436586 - 0.541022256905400i,
            1.04020832976138 + 0.492122906870067i,   0.782150390712463 - 0.386910987957498i,  0.0782189031201583 + 0.414176615986169i, 0.0308962864551090 - 0.0956884023781928i,
            -2.82100726616616 - 0.246639068270180i,  -1.71244139997764 - 0.317978871348536i,  -0.562912629032458 - 1.64731197743026i,  -1.42254228069791 - 1.16950251382209i,
            0.933555188701696 + 0.601148712982981i,  0.948163479538116 - 0.868161159731760i,  0.548150072831567 + 0.526637056017474i,  0.361781340251915 + 0.0750417432243666i
        };
        std::vector<std::complex<double> > A_X_B_ref = 
        {
            -0.483082141821235 - 0.208132660800410i, 1.04020832976138 + 0.492122906870067i,   -2.82100726616616 - 0.246639068270180i,  0.933555188701696 + 0.601148712982981i,
            -0.425092571135247 - 0.918007566698950i, 0.782150390712463 - 0.386910987957498i,  -1.71244139997764 - 0.317978871348536i,  0.948163479538116 - 0.868161159731760i,
            -0.298701024217480 - 0.612324962995601i, 0.0782189031201583 + 0.414176615986169i, -0.562912629032458 - 1.64731197743026i,  0.548150072831567 + 0.526637056017474i,
            -0.469926268436586 - 0.541022256905400i, 0.0308962864551090 - 0.0956884023781928i,    -1.42254228069791 - 1.16950251382209i,   0.361781340251915 + 0.0750417432243666i
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

        blas.double2complex( A.data(), 4, 4, AC.data() );
        blas.gemm(AC.data(), 'N', B.data(), 'N', 4, B1.data() );
        std::cout << "complex(A):" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << AC[j+4*k] << " ";
            }
            std::cout << std::endl;
        } 
        std::cout << "B:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << B[j+4*k] << " ";
            }
            std::cout << std::endl;
        } 
        std::cout << "B1_ref:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << A_X_B_ref[j+4*k] << " ";
            }
            std::cout << std::endl;
        }         
        std::cout << "A*B = B1:" << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << B1[j+4*k] << " ";
            }
            std::cout << std::endl;
        }        
        
        std::cout << "B1-B1_ref: " << std::endl;
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            { 
                std::cout << B1[j+4*k] - A_X_B_ref[j+4*k] << " ";
            }
            std::cout <<  std::endl;
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
