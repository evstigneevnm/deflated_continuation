#ifndef __LAPACK_WRAP_H__
#define __LAPACK_WRAP_H__

/*//
    wrap over some specific LAPACK routines, mainly used for eigenvalue estimation.
*/

#include <utils/cuda_support.h>
#include <cstring>

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

}

template<class T>
class lapack_wrap
{
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

    //LAPACK wrap functions:
    
    //direct upper Hessenberg matrix eigs
    //http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gacb35e85b362ce8ccf9d653cc3f8fb89c.html
    void hessinberg_eigs(const T* H, size_t Nl, T* eig_real, T* eig_imag);
    template<class C_t>
    void hessinberg_eigs(const T* H, size_t Nl, C_t* eig)
    {
        T* eig_real = new T[Nl];
        T* eig_imag = new T[Nl];
        hessinberg_eigs(H, Nl, eig_real, eig_imag);
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }
        delete [] eig_real;
        delete [] eig_imag;        
    }

    //direct upper Hessenberg matrix Shur form, returns R, and Q, s.t.: 
    // H = Q*R*Q^T
    void hessinberg_schur(const T* H, size_t Nl, T* Q, T* R, T* eig_real = nullptr, T* eig_imag = nullptr);
    template<class C_t>
    void hessinberg_schur(const T* H, size_t Nl, T* Q, T* R, C_t* eig)
    {
        T* eig_real = new T[Nl];
        T* eig_imag = new T[Nl];
        hessinberg_schur(H, Nl, Q, R, eig_real, eig_imag);
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }
        delete [] eig_real;
        delete [] eig_imag;        
    }

    // eigenenvalues of a general matrix
    // http://www.netlib.org/lapack/explore-html/d9/d8e/group__double_g_eeigen_ga66e19253344358f5dee1e60502b9e96f.html#ga66e19253344358f5dee1e60502b9e96f
    void eigs(const T* A, size_t Nl, T* eig_real, T* eig_imag);    
    template<class C_t>
    void eigs(const T* A, size_t Nl, C_t* eig)
    {
        T* eig_real = new T[Nl];
        T* eig_imag = new T[Nl];
        eigs(A, Nl, eig_real, eig_imag);
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }
        delete [] eig_real;
        delete [] eig_imag;
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
        T* eig_real = new T[Nl];
        T* eig_imag = new T[Nl];
        T* eigv_R = new T[Nl*Nl];
        eigsv(A, Nl, eig_real, eig_imag, eigv_R);
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
        delete [] eigv_R;
        delete [] eig_real;
        delete [] eig_imag;

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
        T* eig_real = new T[Nl];
        T* eig_imag = new T[Nl];
        hessinberg_eigs_from_gpu(H_device, Nl, eig_real, eig_imag);
        for(int j=0;j<Nl;j++)
        {
            eig[j] = C_t(eig_real[j], eig_imag[j]);
        }
        delete [] eig_real;
        delete [] eig_imag;        
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


    void gemm(const T* A, char opA, const T* B, char opB, size_t Nl, T* C)
    {
        gemm(opA, opB, Nl, Nl, Nl, 1.0, A, B, 0.0, C);
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
    void return_row(size_t row_, const T* A, size_t Nl, T* row)
    {
        for(int j=0;j<Nl;j++)
        {
            row[j] = A[_I2(row_, j, Nl)];
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


  

};



 //LAPACK wrap functions specialization:

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
        throw std::runtime_error("hessinberg_eigs: dhseqr_ returned INFO!=0.");
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
        throw std::runtime_error("hessinberg_eigs: shseqr_ returned INFO!=0.");
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
        throw std::runtime_error("hessinberg_eigs: dhseqr_ returned INFO!=0.");
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
        throw std::runtime_error("hessinberg_eigs: dhseqr_ returned INFO!=0.");
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
        throw std::runtime_error("hessinberg_eigs: shseqr_ returned INFO!=0.");
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
        throw std::runtime_error("eigs: dgeev_ returned INFO!=0.");
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
        throw std::runtime_error("eigsv: dgeev_ returned INFO!=0.");
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
        throw std::runtime_error("eigsv: dgeev_ returned INFO!=0.");
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
#endif