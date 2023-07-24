#ifndef __LAPACK_WRAP_H__
#define __LAPACK_WRAP_H__

/*//
    wrap over some specific LAPACK routines, mainly used for eigenvalue estimation.
*/
#include <complex>
#include <vector>
#include <utils/cuda_support.h>
#include <cstring>
#include <limits>

namespace lapack_exters
{
extern "C" void dhseqr_(char*  JOB,char*  COMPZ, 
                        int* N, int* ILO, 
                        int* IHI,   double *H,
                        int* LDH, 
                        double *WR, double *WI,
                        double *Z, int* LDZ,
                        double* WORK,
                        int* LWORK,int *INFO);

extern "C" void shseqr_(char*  JOB,char*  COMPZ, 
                        int* N, int* ILO, 
                        int* IHI, float *H,
                        int* LDH, 
                        float *WR, float *WI,
                        float *Z, int* LDZ,
                        float* WORK,
                        int* LWORK,int *INFO);

extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a,
              int *lda, double *wR, double *wI,
              double *vl, int *ldvl, double *vr, int *ldvr, 
              double *work, int *lwork, int *info);

extern "C" void cgeev_(char *jobvl, char *jobvr, int *n, float *a,
              int *lda, float *wR, float *wI,
              float *vl, int *ldvl, float *vr, int *ldvr, 
              float *work, int *lwork, int *info);


extern "C"  void dgeqrf_(int* M, int* N, 
                    double* A, int* LDA, double* TAU, 
                    double* WORK, int* LWORK, int* INFO );
extern "C"  void sgeqrf_(int* M, int* N, 
                    float* A, int* LDA, float* TAU, 
                    float* WORK, int* LWORK, int* INFO );

extern "C"  void dormqr_(char*  SIDE, char* TRANS, 
                    int* M, int* N, int* K, 
                    double* A, int* LDA, double* TAU, 
                    double* C, int* LDC,
                    double* WORK, int* LWORK, int* INFO );
extern "C"  void sormqr_(char*  SIDE, char* TRANS, 
                    int* M, int* N, int* K, 
                    float* A, int* LDA, float* TAU, 
                    float* C, int* LDC,
                    float* WORK, int* LWORK, int* INFO );

extern "C"  void dgemm_(
        char* TRANSA,
        char* TRANSB,
        int* M,
        int* N,
        int* K,
        double* ALPHA,
        double* A,
        int* LDA,
        double* B,
        int* LDB,
        double* BETA,
        double* C,
        int* LDC 
    );
extern "C"  void sgemm_(
        char* TRANSA,
        char* TRANSB,
        int* M,
        int* N,
        int* K,
        float* ALPHA,
        float* A,
        int* LDA,
        float* B,
        int* LDB,
        float* BETA,
        float* C,
        int* LDC 
    );
extern "C"  void zgemm_(
        char* TRANSA,
        char* TRANSB,
        int* M,
        int* N,
        int* K,
        std::complex<double>* ALPHA,
        std::complex<double>* A,
        int* LDA,
        std::complex<double>* B,
        int* LDB,
        std::complex<double>* BETA,
        std::complex<double>* C,
        int* LDC
    );
extern "C"  void cgemm_(
        char* TRANSA,
        char* TRANSB,
        int* M,
        int* N,
        int* K,
        std::complex<float>* ALPHA,
        std::complex<float>* A,
        int* LDA,
        std::complex<float>* B,
        int* LDB,
        std::complex<float>* BETA,
        std::complex<float>* C,
        int* LDC
    );
extern "C" void dtrexc_(
        char* COMPQ, 
        int* N, 
        double* T, 
        int* LDT, 
        double* Q, 
        int* LDQ, 
        int* IFST, 
        int* ILST,
        double* WORK, 
        int* INFO );
extern "C" void strexc_(
        char* COMPQ, 
        int* N, 
        float* T, 
        int* LDT, 
        float* Q, 
        int* LDQ, 
        int* IFST, 
        int* ILST,
        float* WORK, 
        int* INFO );
extern "C" void dgees_(
        char* JOBVS,
        char* SORT,
        int* SELECT,
        int* N,
        double* A,
        int* LDA,
        int* SDIM,
        double* WR,
        double* WI,
        double* VS,
        int* LDVS,
        double* WORK,
        int* LWORK,
        int* BWORK,
        int* INFO );
extern "C" void sgees_(
        char* JOBVS,
        char* SORT,
        int* SELECT,
        int* N,
        float* A,
        int* LDA,
        int* SDIM,
        float* WR,
        float* WI,
        float* VS,
        int* LDVS,
        float* WORK,
        int* LWORK,
        int* BWORK,
        int* INFO );
}

template<class T>
class lapack_wrap
{
private:
    T machine_eps_ = std::numeric_limits<T>::epsilon();
public:
    lapack_wrap(size_t expected_size_):
    expected_size(expected_size_)
    {
        set_worker();
        set_helper_matrices(expected_size);
    }
    ~lapack_wrap()
    {
        free_helper_matrices();
        if(worker != NULL)
            free(worker);
    }


    void double2complex(T* D, size_t cols, size_t rows, std::complex<T>* C)const
    {   
        for(size_t j=0;j<rows*cols;j++)
        {
            C[j] = static_cast< std::complex<T> >(D[j]);
        }
    }
    void complex2double(std::complex<T>* C, size_t cols, size_t rows, T* D)const
    {   
        for(size_t j=0;j<rows*cols;j++)
        {
            D[j] = C[j].real();
        }
    }


//   takes a quasitriangular Schur matrix R (typically produced by hessinberg_schur=>dhseqr_) 
//   and returns eigenvalues in their order of appearance down the block diagonal of R. 
//   self made port to cpp from matlab
    template<class C_t>
    void schur_upper_triag_ordered_eigs(const T* R, size_t Nl,  C_t* eig)const
    {
        

        std::vector<bool> subdiag(Nl);
        for(size_t l=0;l<Nl-1;l++)
        {
            subdiag.at(l) = std::abs(R[I2_R(l+1,l,Nl)])>machine_eps_;
        }
        size_t ind_m = 0, ind_p = 0, eig_ind = 0;
        while(true)
        {
            std::vector<T> submatrix;
            ind_p += subdiag.at(ind_p)?1:0;
            for(int j=ind_m;j<=ind_p;j++)
            {
                for(int k=ind_m;k<=ind_p;k++)
                {
                    submatrix.push_back( R[I2_R(j,k,Nl)] );
                    // std::cout << submatrix.at(I2_R(j-ind_m,k-ind_m,2)) << " ";
                }
                // std::cout << std::endl;
            }
            // std::cout << " --- " << std::endl;
            auto ret_eig = solve_2_2_eigensystem(submatrix);
            if(ret_eig.size() == 1)
            {
                eig[eig_ind++] = ret_eig.at(0);
            }
            else
            {
                eig[eig_ind++] = ret_eig.at(0);
                eig[eig_ind++] = ret_eig.at(1);
            }

            ++ind_p;
            ind_m = ind_p;
            if(ind_p>=Nl)
            {
                break;
            }          
        }


    }


    //LAPACK wrap functions:
    
    //direct upper Hessenberg matrix eigs
    //http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gacb35e85b362ce8ccf9d653cc3f8fb89c.html
    void hessinberg_eigs(const T* H, size_t Nl, T* eig_real, T* eig_imag);
    template<class C_t>
    void hessinberg_eigs(const T* H, size_t Nl, C_t* eig)
    {
        std::vector<T> eig_real(Nl);
        std::vector<T> eig_imag(Nl);
        hessinberg_eigs(H, Nl, eig_real.data(), eig_imag.data() );
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }       
    }

    //reorder the real Schur factorization of a real matrix Q*R*Q^T, so that the diagonal block
    // of T with row index IFST is moved to row ILST 
    // input matrices are overwritten
    void reorder_schur(T* R, T* Q, size_t Nl, size_t ifst_row, size_t ilst_row);


    //direct upper Hessenberg matrix Shur form, returns R, and Q, s.t.: 
    // H = Q*R*Q^T
    void hessinberg_schur(const T* H, size_t Nl, T* Q, T* R, T* eig_real = nullptr, T* eig_imag = nullptr);
    template<class C_t>
    void hessinberg_schur(const T* H, size_t Nl, T* Q, T* R, C_t* eig)
    {
        std::vector<T> eig_real(Nl);
        std::vector<T> eig_imag(Nl);
        hessinberg_schur(H, Nl, Q, R, eig_real.data(), eig_imag.data() );
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }      
    }

    // eigenenvalues of a general matrix
    // http://www.netlib.org/lapack/explore-html/d9/d8e/group__double_g_eeigen_ga66e19253344358f5dee1e60502b9e96f.html#ga66e19253344358f5dee1e60502b9e96f
    void eigs(const T* A, size_t Nl, T* eig_real, T* eig_imag);    
    template<class C_t>
    void eigs(const T* A, size_t Nl, C_t* eig)
    {
        std::vector<T> eig_real(Nl);
        std::vector<T> eig_imag(Nl);
        eigs(A, Nl, eig_real.data(), eig_imag.data());
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }
    }
    // eigenv*s of a general matrix
    void eigsv(const T* A, size_t Nl, T* eig_real, T* eig_imag, T* eigv_R);    
    template<class C_t>
    void eigsv(const T* A, size_t Nl, C_t* eig, C_t* eigv)
    {
      // If the j-th eigenvalue is real, then v(j) = VR(:,j),
      // the j-th column of VR.
      // If the j-th and (j+1)-st eigenvalues form a complex
      // conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
      // v(j+1) = VR(:,j) - i*VR(:,j+1).                
        std::vector<T> eig_real(Nl,0);
        std::vector<T> eig_imag(Nl,0);
        std::vector<T> eigv_R(Nl*Nl,0);
        eigsv(A, Nl, eig_real.data(), eig_imag.data(), eigv_R.data());
        for(int k=0;k<Nl;k++)
        {
            eig[k] = C_t(eig_real[k], eig_imag[k]);
        }
        int k=0;
        while(k<Nl)
        {
            if(eig_imag[k] == 0)
            {
                for(int j=0;j<Nl;j++)
                {
                    eigv[j + Nl*k] = C_t(eigv_R[j + Nl*k],0);
                }
                k++;
            }
            else
            {
                for(int j=0;j<Nl;j++)
                {
                    eigv[j + Nl*k] = C_t(eigv_R[j + Nl*k],eigv_R[j + Nl*(k+1)]);
                    eigv[j + Nl*(k+1)] = C_t(eigv_R[j + Nl*k],-eigv_R[j + Nl*(k+1)]);
                }
                k+=2;                
            }
        }

    }    

    void eigs_schur(const T* A, size_t Nl, T* eigs_real, T* eigs_imag, T* Q, T* R);

    template<class Ct>
    void eigs_schur(const T* A, size_t Nl,  Ct* eigs, T* Q, T* R)
    {
        std::vector<T> eig_real(Nl,0);
        std::vector<T> eig_imag(Nl,0);
        eigs_schur(A, Nl, eig_real.data(), eig_imag.data(), Q, R);
        for(size_t j=0;j<Nl;j++)
        {
            eigs[j] = Ct(eig_real[j], eig_imag[j]);
        }

    }

    //direct upper Hessenberg matrix eigs from the device
    void hessinberg_eigs_from_gpu(const T* H_device, size_t Nl, T* eig_real, T* eig_imag)
    {

        device_2_host_cpy<T>(A_, (T*)H_device, Nl*Nl);
        hessinberg_eigs(A_, Nl, eig_real, eig_imag);
    }
    template<class C_t>
    void hessinberg_eigs_from_gpu(const T* H_device, size_t Nl, C_t* eig)
    {
        std::vector<T> eig_real(Nl);
        std::vector<T> eig_imag(Nl);
        hessinberg_eigs_from_gpu(H_device, Nl, eig_real.data(), eig_imag.data() );
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }        
    }
    void hessinberg_schur_from_gpu(const T* H_device, size_t Nl, T* Q, T* R, T* eig_real = nullptr, T* eig_imag = nullptr)
    {
        device_2_host_cpy<T>(A_, (T*)H_device, Nl*Nl);
        hessinberg_schur(A_, Nl, Q, R, eig_real, eig_imag);
    }
    template<class C_t>
    void hessinberg_schur_from_gpu(const T* H_device, size_t Nl, T* Q, T* R, C_t* eig)
    {
        device_2_host_cpy<T>(A_, (T*)H_device, Nl*Nl);
        hessinberg_schur(A_, Nl, Q, R, eig);
    }
    void eigs_schur_from_gpu(const T* A_device, size_t Nl, T* Q, T* R, T* eig_real = nullptr, T* eig_imag = nullptr)
    {
        device_2_host_cpy<T>(A_, (T*)A_device, Nl*Nl);
        eigs_schur(A_, Nl, eig_real, eig_imag, Q, R);
    }
    template<class C_t>
    void eigs_schur_from_gpu(const T* A_device, size_t Nl, T* Q, T* R, C_t* eig)
    {
        std::vector<T> A_l(Nl*Nl);

        device_2_host_cpy<T>(A_l.data(), (T*)A_device, Nl*Nl);
        eigs_schur(A_l.data(), Nl, eig, Q, R);
    }



    void qr(const T* H, size_t Nl, T* Q, T* R = nullptr)
    {
        
        set_helper_matrices(Nl);
        set_worker();
        copy_square_1_to_2(H, A_, Nl);
        if(R == nullptr)
        {
            qr_no_R_(A_, Q);
        }
        else
        {
            qr_(A_, Q, R);
        }
    }

    //LDA = max( 1, m ) for opA='N' and LDA = max(1,k) otherwise
    //LDB = max( 1, k ) for opB='N' and LDB =max( 1, n ) otherwise
    //LDC = max( 1, m )
    void gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const T alpha, const T* A, const T* B, const T beta, T* C);

    void gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const std::complex<T> alpha, const std::complex<T>* A, const std::complex<T>* B, const std::complex<T> beta, std::complex<T>* C);

    void gemm(const T* A, char opA, const T* B, char opB, size_t Nl, T* C)
    {
        gemm(opA, opB, Nl, Nl, Nl, 1.0, A, B, 0.0, C);
    }
    void gemm(const std::complex<T>* A, char opA, const std::complex<T>* B, char opB, size_t Nl, std::complex<T>* C)
    {
        gemm(opA, opB, Nl, Nl, Nl, std::complex<T>(1.0), A, B, std::complex<T>(0.0), C);
    }


    void mat_sq(const T* A, size_t Nl, T* A2)
    {
        gemm(A, 'N', A, 'N', Nl, A2);
    }

    void add_to_diagonal(const T val, size_t Nl, T* A)
    {
        for(int j=0;j<Nl;j++)
        {
            A[_I2(j, j, Nl)]+=val;
        }

    }


    void return_col(size_t col_, const T* A, size_t Nl, T* col)
    {
        for(int j=0;j<Nl;j++)
        {
            col[j] = A[_I2(j, col_, Nl)];
        }
    }
    void set_col(size_t col_p, const T* col, T* A, size_t Nl, size_t col_size = 0)
    {
        if(col_size == 0)
        {
            col_size = Nl;
        }
        for(int j=0;j<col_size;j++)
        {
            A[_I2(j, col_p, Nl)] = col[j];
        }
    }    
    void return_row(size_t row_, const T* A, size_t Nl, T* row)
    {
        for(int j=0;j<Nl;j++)
        {
            row[j] = A[_I2(row_, j, Nl)];
        }
    }
    void set_row(size_t row_p, const T* row, T* A, size_t Nl, size_t row_size = 0)
    {
        if(row_size == 0)
        {
            row_size = Nl;
        }
        for(int j=0;j<row_size;j++)
        {
            A[_I2(row_p, j, Nl)] = row[j];
        }
    }    
    void return_col(size_t col_, const std::complex<T>* A, size_t Nl, std::complex<T>* col)
    {
        for(int j=0;j<Nl;j++)
        {
            col[j] = A[_I2(j, col_, Nl)];
        }
    }
    void return_row(size_t row_, const std::complex<T>* A, size_t Nl, std::complex<T>* row)
    {
        for(int j=0;j<Nl;j++)
        {
            row[j] = A[_I2(row_, j, Nl)];
        }
    }

    // std::pair<size_t, size_t> (row, col)
    void set_submatrix(std::pair<size_t, size_t> start, const T* subA, std::pair<size_t, size_t> subsize, T* A, std::pair<size_t, size_t> size )
    {
        size_t row_start = start.first;
        size_t col_start = start.second;
        size_t row_small_size = subsize.first;
        size_t col_small_size = subsize.second;        
        size_t rows = size.first;
        size_t cols = size.second;

        for(size_t j=row_start;j<row_small_size+row_start;j++)
        {
            for(size_t k=col_start;k<col_small_size+col_start;k++)
            {
                auto a = subA[_I2(j-row_start, k-col_start, row_small_size) ];
                A[_I2(j, k, rows)] = a;
            }

        }

    }
    // std::pair<size_t, size_t> (row, col)
    void return_submatrix(const T* A, std::pair<size_t, size_t> size, std::pair<size_t, size_t> start, T* subA, std::pair<size_t, size_t> subsize)
    {
        size_t row_start = start.first;
        size_t col_start = start.second;
        size_t row_small_size = subsize.first;
        size_t col_small_size = subsize.second;        
        size_t rows = size.first;
        size_t cols = size.second;

        for(size_t j=row_start;j<row_small_size+row_start;j++)
        {
            for(size_t k=col_start;k<col_small_size+col_start;k++)
            {
                auto sub_j = j-row_start;
                auto sub_k = k-col_start;
                auto a = A[_I2(j, k, rows)];
                subA[_I2(sub_j, sub_k, row_small_size) ] = a;
            }

        }

    }

    template<class T_l>
    void eye(T_l* A)
    {
        eye<T_l>(A, expected_size);
    }  

    template<class T_l>
    void eye(T_l* A, size_t Nl)
    {
        for(int j = 0; j<Nl; j++)
        {
            for(int k=0; k<Nl; k++)
            {
                A[_I2(j,k, Nl)] = T_l(0);
            }
            A[_I2(j,j, Nl)] = T_l(1);
        }
    } 
    
    template<class T_l>
    void copy_square_1_to_2(const T_l* A, T_l* B, size_t sz_)
    {
        if(A != B)
        {
            size_t sz_bytes_ = sizeof(T_l)*sz_*sz_;
            std::memcpy(B, A, sz_bytes_);
        }
    }

    template <class T_l>
    void write_matrix(const std::string &f_name, size_t Row, size_t Col, T_l *matrix, unsigned int prec=17)
    {
        size_t N=Col;
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
        for (size_t i = 0; i<Row; i++)
        {
            for(size_t j=0;j<Col;j++)
            {
                if(j<Col-1)
                    f << std::setprecision(prec) << matrix[I2_R(i,j,Row)] << " ";
                else
                    f << std::setprecision(prec) << matrix[I2_R(i,j,Row)];

            }
            f << std::endl;
        } 
        
        f.close();
    }
    template <class T_l>
    void write_vector(const std::string &f_name, size_t N, T_l *vec, unsigned int prec=17)
    {
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
        for (size_t i = 0; i<N; i++)
        {
            f << std::setprecision(prec) << vec[i] << std::endl;
        } 
        f.close();
    }    


    template<class T_l>
    void write_matrix_from_device(const std::string &f_name, size_t Row, size_t Col, T_l *matrix, unsigned int prec=17)
    {
        std::vector<T_l> A_l(Row*Col, 0);
        device_2_host_cpy<T_l>(A_l.data(), (T_l*)matrix, Row*Col);
        write_matrix(f_name, Row, Col, A_l.data(), prec);
    }

    template<class T_l>
    void write_vector_from_device(const std::string &f_name, size_t N, T_l *vec, unsigned int prec=17)
    {
        std::vector<T_l> v_l(N,0);
        device_2_host_cpy<T_l>(v_l.data(), (T_l*)vec, N);
        write_vector(f_name, N, v_l, prec);
    }

private:
    void qr_no_R_(T* A, T* Q);
    void qr_(T* A, T* Q, T* R);

    T* worker = NULL;
    T* A_ = NULL;
    T* B_ = NULL;

    const int worker_coeff = 11;
    size_t expected_size;

    void set_worker()
    {
        if(worker == NULL)
        {
            worker = (T*) malloc(worker_coeff*expected_size*sizeof(T));
            if(worker == NULL)
                throw std::runtime_error("lapack_wrap: filed to allocate worker array.");
        }
        else
        {
            //realloc, but for some weird reason it's unstable?!?
            free(worker);
            worker = NULL;
            set_worker();             
        }
    }
    void set_helper_matrices(size_t sz_)
    {
        
        if(sz_ > expected_size)
        {
            expected_size = sz_;
            free_helper_matrices();
            set_helper_matrices(expected_size);
        }
        if(A_ == NULL)
        {
            A_ = (T*) malloc(expected_size*expected_size*sizeof(T));
            if(A_ == NULL)
                throw std::runtime_error("lapack_wrap: filed to allocate helper matrix A_.");
        }
        if(B_ == NULL)
        {
            B_ = (T*) malloc(expected_size*expected_size*sizeof(T));
            if(B_ == NULL)
                throw std::runtime_error("lapack_wrap: filed to allocate helper matrix B_.");
        }
    }
    void free_helper_matrices()
    {
        if(A_ != NULL)
        {
            free(A_);
            A_ = NULL;
        }
        if(B_ != NULL)
        {
            free(B_);
            B_ = NULL;
        }        
    }



    //column majour format:
    // j - row numner
    // k - column number
    size_t inline _I2(int j, int k, size_t Nl) 
    {
        return(j + Nl*k);
    }
    size_t inline _I2(int j, int k) 
    {
        return _I2(j, k, expected_size);
    }

    std::vector< std::complex<T> > solve_2_2_eigensystem(std::vector< T >& submatrix)const
    {
        if(submatrix.size() == 1)       
        {
            std::vector<std::complex<T> > solution{submatrix.at(0)};
            return solution;
        }
        else
        {
            T a11 = submatrix.at( I2_R(0,0,2) );
            T a12 = submatrix.at( I2_R(0,1,2) );
            T a21 = submatrix.at( I2_R(1,0,2) );
            T a22 = submatrix.at( I2_R(1,1,2) );
            T b = -(a11+a22);
            T c = (a11*a22-a12*a21);
            T D = b*b-4.0*c;
            if(D>=0.0)
            {
                throw std::runtime_error("lapack_wrap::solve_2_2_eigensystem: real eigenvalue for a 2X2 Schur block.");
            }
            T sqrtD = std::sqrt(-D);
            T realL = -0.5*b;
            T imagL = 0.5*sqrtD;
            std::vector< std::complex<T> > solution;
            solution.emplace_back(realL,+imagL);
            solution.emplace_back(realL,-imagL);

            return solution;
        }

    }

  

};



 //LAPACK wrap functions specialization:


template<> inline
void lapack_wrap<double>::reorder_schur(double* R, double* Q, size_t Nl, size_t ifst_row, size_t ilst_row)
{
    set_helper_matrices(Nl);
    char COMPQ = 'V'; //update Q matrix
    int N = Nl;
    int LDR = Nl;
    int LDQ = Nl;
    int IFST = ifst_row;
    int ILST = ilst_row;
    int INFO = 0;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }
      // SUBROUTINE DTREXC( COMPQ, N, T, LDT, Q, LDQ, IFST, ILST,
      //                    WORK, INFO )

      //     CHARACTER      COMPQ

      //     INTEGER        IFST, ILST, INFO, LDQ, LDT, N

      //     DOUBLE         PRECISION Q( LDQ, * ), T( LDT, * ), WORK(
      //                    * )    
    lapack_exters::dtrexc_(&COMPQ, &N, R, &LDR, Q, &LDQ, &IFST, &ILST, worker, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("reorder_schur: dtrexc_ returned INFO!=0:" + std::to_string(INFO) );
    }    

}

template<> inline
void lapack_wrap<float>::reorder_schur(float* R, float* Q, size_t Nl, size_t ifst_row, size_t ilst_row)
{
    set_helper_matrices(Nl);
    char COMPQ = 'V'; //update Q matrix
    int N = Nl;
    int LDR = Nl;
    int LDQ = Nl;
    int IFST = ifst_row;
    int ILST = ilst_row;
    int INFO = 0;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }
      // SUBROUTINE DTREXC( COMPQ, N, T, LDT, Q, LDQ, IFST, ILST,
      //                    WORK, INFO )

      //     CHARACTER      COMPQ

      //     INTEGER        IFST, ILST, INFO, LDQ, LDT, N

      //     DOUBLE         PRECISION Q( LDQ, * ), T( LDT, * ), WORK(
      //                    * )    
    lapack_exters::strexc_(&COMPQ, &N, R, &LDR, Q, &LDQ, &IFST, &ILST, worker, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("reorder_schur: strexc_ returned INFO!=0:" + std::to_string(INFO) );
    }    

}

template<> inline
void lapack_wrap<double>::hessinberg_eigs(const double* H, size_t Nl, double* eig_real, double* eig_imag)
{
    set_helper_matrices(Nl);
    copy_square_1_to_2(H, A_, Nl);
    char JOB='E';
    char COMPZ='N';
    int N = Nl;
    int ILO = 1;
    int IHI = Nl;
    int LDH = Nl;
    double *Z = NULL;
    int LDZ = 1;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }

    int LWORK = worker_coeff*expected_size;
    int INFO = 0;

    lapack_exters::dhseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, A_,
            &LDH, 
            eig_real, eig_imag,
            Z, &LDZ,
            worker,
            &LWORK, &INFO);
    
    if(INFO!=0)
    {
        throw std::runtime_error("hessinberg_eigs: dhseqr_ returned INFO!=0:" + std::to_string(INFO) );
    }

}

template<> inline
void lapack_wrap<float>::hessinberg_eigs(const float* H, size_t Nl, float* eig_real, float* eig_imag)
{
    set_helper_matrices(Nl);
    copy_square_1_to_2(H, A_, Nl);
    char JOB='E';
    char COMPZ='N';
    int N = Nl;
    int ILO = 1;
    int IHI = Nl;
    int LDH = Nl;
    float *Z = NULL;
    int LDZ = 1;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }

    int LWORK = worker_coeff*expected_size;
    int INFO = 0;

    lapack_exters::shseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, A_,
            &LDH, 
            eig_real, eig_imag,
            Z, &LDZ,
            worker,
            &LWORK, &INFO);
    
    if(INFO!=0)
    {
        throw std::runtime_error("hessinberg_eigs: shseqr_ returned INFO!=0: " + std::to_string(INFO) );
    }

}




template<> inline
void lapack_wrap<double>::hessinberg_schur(const double* H, size_t Nl, double* Q, double* R, double* eig_real, double* eig_imag)
{
    set_helper_matrices(Nl);
    copy_square_1_to_2(H, A_, Nl);
    char JOB='S';
    char COMPZ='I';
    int N = Nl;
    int ILO = 1;
    int IHI = Nl;
    int LDH = Nl;
    int LDZ = Nl;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }

    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    bool use_internal_eigs = true;
    if((eig_real != nullptr)||(eig_imag != nullptr))
    {
        use_internal_eigs = false;
    }
    if(use_internal_eigs)
    {
        eig_real = new double[N];
        eig_imag = new double[N];
    }

    lapack_exters::dhseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, A_,
            &LDH, 
            eig_real, eig_imag,
            Q, &LDZ,
            worker,
            &LWORK, &INFO);
    
    copy_square_1_to_2(A_, R, Nl);
    if(INFO!=0)
    {
        throw std::runtime_error("hessinberg_schur: dhseqr_ returned INFO!=0:" + std::to_string(INFO) );
    }
    
    if(use_internal_eigs)
    {
        delete [] eig_real;
        delete [] eig_imag;
    }

}
template<> inline
void lapack_wrap<float>::hessinberg_schur(const float* H, size_t Nl, float* Q, float* R, float* eig_real, float* eig_imag)
{
    set_helper_matrices(Nl);
    copy_square_1_to_2(H, A_, Nl);
    char JOB='S';
    char COMPZ='I';
    int N = Nl;
    int ILO = 1;
    int IHI = Nl;
    int LDH = Nl;
    int LDZ = Nl;
    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }

    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    bool use_internal_eigs = true;
    if((eig_real != nullptr)&&(eig_imag != nullptr))
    {
        use_internal_eigs = false;
    }
    if(use_internal_eigs)
    {
        eig_real = new float[N];
        eig_imag = new float[N];
    }

    lapack_exters::shseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, A_,
            &LDH, 
            eig_real, eig_imag,
            Q, &LDZ,
            worker,
            &LWORK, &INFO);
    
    copy_square_1_to_2(A_, R, Nl);
    if(INFO!=0)
    {
        throw std::runtime_error("hessinberg_eigs: shseqr_ returned INFO!=0: " + std::to_string(INFO) );
    }
    if(use_internal_eigs)
    {
        delete [] eig_real;
        delete [] eig_imag;
    }

}


template<> inline
void lapack_wrap<double>::eigs(const double* A, size_t Nl, double* eig_real, double* eig_imag)
{

    set_helper_matrices(Nl);
    copy_square_1_to_2(A, A_, Nl);

    char JOBVL ='N';   // Compute Right eigenvectors
    char JOBVR ='N';   // Do not compute Left eigenvectors
    int N = Nl;
    int LDVL = 1; 
    int LDVR = 1;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    double * VL = NULL;
    double * VR = NULL;
   
    lapack_exters::dgeev_( &JOBVL, &JOBVR, &N, A_,  &N, eig_real,
            eig_imag, VL, &LDVL,
            VR, &LDVR, 
            worker, 
            &LWORK, &INFO );

     if(INFO!=0)
    {
        throw std::runtime_error("hessinberg_eigs: shseqr_ returned INFO!=0: " + std::to_string(INFO) );
    }  

}


template<> inline
void lapack_wrap<float>::eigs(const float* A, size_t Nl, float* eig_real, float* eig_imag)
{

    set_helper_matrices(Nl);
    copy_square_1_to_2(A, A_, Nl);
    char JOBVL ='N';   // Compute Right eigenvectors
    char JOBVR ='N';   // Do not compute Left eigenvectors
    int N = Nl;
    int LDVL = 1; 
    int LDVR = 1;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    float * VL = NULL;
    float * VR = NULL;
   
    lapack_exters::cgeev_( &JOBVL, &JOBVR, &N, A_,  &N, eig_real,
            eig_imag, VL, &LDVL,
            VR, &LDVR, 
            worker, 
            &LWORK, &INFO );

     if(INFO!=0)
    {
        throw std::runtime_error("eigs: dgeev_ returned INFO!=0: " + std::to_string(INFO) );
    }  

}


template<> inline
void lapack_wrap<double>::eigsv(const double* A, size_t Nl, double* eig_real, double* eig_imag, double* eigv_R)
{

    set_helper_matrices(Nl);
    copy_square_1_to_2(A, A_, Nl);
    char JOBVL ='N';   // Do not compute Left eigenvectors
    char JOBVR ='V';   // Compute Right eigenvectors
    int N = Nl;
    int LDVL = 1; 
    int LDVR = N;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    double * VL = NULL;
    double * VR = eigv_R;
   
    lapack_exters::dgeev_( &JOBVL, &JOBVR, &N, A_,  &N, eig_real,
            eig_imag, VL, &LDVL,
            VR, &LDVR, 
            worker, 
            &LWORK, &INFO );

     if(INFO!=0)
    {
        throw std::runtime_error("eigsv: dgeev_ returned INFO!=0:" + std::to_string(INFO) );
    }  

}
template<> inline
void lapack_wrap<float>::eigsv(const float* A, size_t Nl, float* eig_real, float* eig_imag, float* eigv_R)
{

    set_helper_matrices(Nl);
    copy_square_1_to_2(A, A_, Nl);

    char JOBVL ='N';   // Do not compute Left eigenvectors
    char JOBVR ='V';   // Compute Right eigenvectors
    int N = Nl;
    int LDVL = 1; 
    int LDVR = N;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
    int INFO = 0;
    float * VL = NULL;
    float * VR = eigv_R;
   
    lapack_exters::cgeev_( &JOBVL, &JOBVR, &N, A_,  &N, eig_real,
            eig_imag, VL, &LDVL,
            VR, &LDVR, 
            worker, 
            &LWORK, &INFO );

     if(INFO!=0)
    {
        throw std::runtime_error("eigsv: dgeev_ returned INFO!=0: " + std::to_string(INFO) );
    }  

}


template<> inline
void lapack_wrap<double>::qr_(double* A, double* Q, double* R)
{
    int N = expected_size;
    int M=N,LDA=N;
    int INFO=0;
    int LWORK = worker_coeff*N;


    lapack_exters::dgeqrf_(&M, &N, A, &LDA, B_, worker, &LWORK, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("qr_: dgeqrf_ argument " + std::to_string(INFO) + " has an illegal value.");
    }




    copy_square_1_to_2(A, R, N);
    
    for(int j=0;j<N-1;j++){
        for(int i=N-1;i>j;i--){
            R[_I2(i,j)]=0.0;   //remove (v)-s below diagonal
        }
    }
    

    char SIDE='L';
    char TRANS='N'; //transposed output
    int K=N, LDC=N;
    eye(Q); 
    lapack_exters::dormqr_(&SIDE, &TRANS, &M, &N, &K, A, &LDA, B_, Q, &LDC,
                    worker, &LWORK, &INFO );
    if(INFO!=0)
    {
        throw std::runtime_error("qr_: dormqr_ argument " + std::to_string(INFO) + " has an illegal value.");
    }

}
template<> inline
void lapack_wrap<float>::qr_(float* A, float* Q, float* R)
{
    int N = expected_size;
    int M=N,LDA=N;
    int INFO=0;
    int LWORK = worker_coeff*N;


    lapack_exters::sgeqrf_(&M, &N, A, &LDA, B_, worker, &LWORK, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("qr_: sgeqrf_ argument " + std::to_string(INFO) + " has an illegal value.");
    }




    copy_square_1_to_2(A, R, N);
    
    for(int j=0;j<N-1;j++){
        for(int i=N-1;i>j;i--){
            R[_I2(i,j)]=0.0;   //remove (v)-s below diagonal
        }
    }
    

    char SIDE='L';
    char TRANS='N'; //transposed output
    int K=N, LDC=N;
    eye(Q); 
    lapack_exters::sormqr_(&SIDE, &TRANS, &M, &N, &K, A, &LDA, B_, Q, &LDC,
                    worker, &LWORK, &INFO );
    if(INFO!=0)
    {
        throw std::runtime_error("qr_: sormqr_ argument " + std::to_string(INFO) + " has an illegal value.");
    }

}


template<> inline
void lapack_wrap<double>::qr_no_R_(double* A, double* Q)
{
    int N = expected_size;
    int M=N,LDA=N;
    int INFO=0;
    int LWORK = worker_coeff*N;


    lapack_exters::dgeqrf_(&M, &N, A, &LDA, B_, worker, &LWORK, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("qr_no_R_: dgeqrf_ argument " + std::to_string(INFO) + " has an illegal value.");
    }
    


    char SIDE='R';
    char TRANS='N'; //transposed output
    int K=N, LDC=N;
    eye(Q); 
    lapack_exters::dormqr_(&SIDE, &TRANS, &M, &N, &K, A, &LDA, B_, Q, &LDC,
                    worker, &LWORK, &INFO );
    if(INFO!=0)
    {
        throw std::runtime_error("qr_no_R_: dormqr_ argument " + std::to_string(INFO) + " has an illegal value.");
    }

}
template<> inline
void lapack_wrap<float>::qr_no_R_(float* A, float* Q)
{
    int N = expected_size;
    int M=N,LDA=N;
    int INFO=0;
    int LWORK = worker_coeff*N;


    lapack_exters::sgeqrf_(&M, &N, A, &LDA, B_, worker, &LWORK, &INFO);
    if(INFO!=0)
    {
        throw std::runtime_error("qr_no_R_: dgeqrf_ argument " + std::to_string(INFO) + " has an illegal value.");
    }
    

    char SIDE='L';
    char TRANS='N'; //transposed output
    int K=N, LDC=N;
    eye(Q); 
    lapack_exters::sormqr_(&SIDE, &TRANS, &M, &N, &K, A, &LDA, B_, Q, &LDC,
                    worker, &LWORK, &INFO );
    if(INFO!=0)
    {
        throw std::runtime_error("qr_no_R_: dormqr_ argument " + std::to_string(INFO) + " has an illegal value.");
    }

}


    //LDA = max( 1, m ) for opA='N' and LDA = max(1,k) otherwise
    //LDB = max( 1, k ) for opB='N' and LDB =max( 1, n ) otherwise
    //LDC = max( 1, m )
template<> inline
void lapack_wrap<double>::gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const double alpha, const double* A, const double* B, const double beta, double* C)
{

    char TRANSA = opA;
    char TRANSB = opB;
    int M = rows_op_A;
    int N = cols_op_B;
    int K = cols_op_A_rows_op_B;
    double ALPHA = alpha;
    int LDA = M;
    if(TRANSA != 'N') LDA = K;
    int LDB = K;
    if(TRANSB != 'N') LDB = N;
    double BETA = beta;
    int LDC = M;

    lapack_exters::dgemm_(
        &TRANSA,
        &TRANSB,
        &M,
        &N,
        &K,
        &ALPHA,
        (double*)A,
        &LDA,
        (double*)B,
        &LDB,
        &BETA,
        C,
        &LDC 
    );

}
template<> inline
void lapack_wrap<float>::gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const float alpha, const float* A, const float* B, const float beta, float* C)
{

    char TRANSA = opA;
    char TRANSB = opB;
    int M = rows_op_A;
    int N = cols_op_B;
    int K = cols_op_A_rows_op_B;
    float ALPHA = alpha;
    int LDA = M;
    if(TRANSA != 'N') LDA = K;
    int LDB = K;
    if(TRANSB != 'N') LDB = N;
    float BETA = beta;
    int LDC = M;

    lapack_exters::sgemm_(
        &TRANSA,
        &TRANSB,
        &M,
        &N,
        &K,
        &ALPHA,
        (float*)A,
        &LDA,
        (float*)B,
        &LDB,
        &BETA,
        C,
        &LDC 
    );

}
template<> inline
void lapack_wrap<double>::gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const std::complex<double> alpha, const std::complex<double>* A, const std::complex<double>* B, const std::complex<double> beta, std::complex<double>* C)
{

    char TRANSA = opA;
    char TRANSB = opB;
    int M = rows_op_A;
    int N = cols_op_B;
    int K = cols_op_A_rows_op_B;
    std::complex<double> ALPHA = alpha;
    int LDA = M;
    if(TRANSA != 'N') LDA = K;
    int LDB = K;
    if(TRANSB != 'N') LDB = N;
    std::complex<double> BETA = beta;
    int LDC = M;

    lapack_exters::zgemm_(
        &TRANSA,
        &TRANSB,
        &M,
        &N,
        &K,
        &ALPHA,
        (std::complex<double>*)A,
        &LDA,
        (std::complex<double>*)B,
        &LDB,
        &BETA,
        C,
        &LDC 
    );

}
template<> inline
void lapack_wrap<float>::gemm(char opA, char opB, int rows_op_A, int cols_op_B, int cols_op_A_rows_op_B, const std::complex<float> alpha, const std::complex<float>* A, const std::complex<float>* B, const std::complex<float> beta, std::complex<float>* C)
{

    char TRANSA = opA;
    char TRANSB = opB;
    int M = rows_op_A;
    int N = cols_op_B;
    int K = cols_op_A_rows_op_B;
    std::complex<float> ALPHA = alpha;
    int LDA = M;
    if(TRANSA != 'N') LDA = K;
    int LDB = K;
    if(TRANSB != 'N') LDB = N;
    std::complex<float> BETA = beta;
    int LDC = M;

    lapack_exters::cgemm_(
        &TRANSA,
        &TRANSB,
        &M,
        &N,
        &K,
        &ALPHA,
        (std::complex<float>*)A,
        &LDA,
        (std::complex<float>*)B,
        &LDB,
        &BETA,
        C,
        &LDC 
    );

}

template<> inline
void lapack_wrap<double>::eigs_schur(const double* A, size_t Nl, double* eigs_real, double* eigs_imag, double* Q, double* R)
{

    copy_square_1_to_2(A, R, Nl);
    char JOBVS ='V';  //Schur vectors are computed.
    char SORT ='N'; //Eigenvalues are not ordered;
    int SELECT = 0;
    int N = Nl;
    int SDIM = 0;
    int BWORK = 0;
    int INFO = 0;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
   
    lapack_exters::dgees_(
        &JOBVS,
        &SORT,
        &SELECT,
        &N,
        R,
        &N,
        &SDIM,
        eigs_real,
        eigs_imag,
        Q,
        &N,
        worker,
        &LWORK,
        &BWORK,
        &INFO
    );

     if(INFO!=0)
    {
        throw std::runtime_error("eigs_schur: dgees_ returned INFO!=0:" + std::to_string(INFO) );
    }  

}
template<> inline
void lapack_wrap<float>::eigs_schur(const float* A, size_t Nl, float* eigs_real, float* eigs_imag, float* Q, float* R)
{

    copy_square_1_to_2(A, R, Nl);
    char JOBVS ='V';  //Schur vectors are computed.
    char SORT ='N'; //Eigenvalues are not ordered;
    int SELECT = 0;
    int N = Nl;
    int SDIM = 0;
    int BWORK = 0;
    int INFO = 0;

    if(expected_size < Nl)
    {
        expected_size = Nl;
        set_worker();
    }


    int LWORK = worker_coeff*expected_size;
   
    lapack_exters::sgees_(
        &JOBVS,
        &SORT,
        &SELECT,
        &N,
        R,
        &N,
        &SDIM,
        eigs_real,
        eigs_imag,
        Q,
        &N,
        worker,
        &LWORK,
        &BWORK,
        &INFO
    );

     if(INFO!=0)
    {
        throw std::runtime_error("eigs_schur: sgees_ returned INFO!=0:" + std::to_string(INFO) );
    }  

}




#endif