#ifndef __LAPACK_WRAP_H__
#define __LAPACK_WRAP_H__

/*//
    wrap over some specific LAPACK routines, mainly used for eigenvalue estimation.
*/

#include <utils/cuda_support.h>


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


template<class T>
class lapack_wrap
{
public:
    lapack_wrap(size_t expected_size_):
    expected_size(expected_size_)
    {
        set_worker();
    }
    ~lapack_wrap()
    {
        if(worker != NULL)
            free(worker);
    }


    //LAPACK wrap functions:
    //direct upper Hessenberg matrix eigs
    void hessinberg_eigs(T* H, size_t Nl, T* eig_real, T* eig_imag);
    
    //direct upper Hessenberg matrix eigs from the device
    void hessinberg_eigs_from_gpu(const T* H_device, size_t Nl, T* eig_real, T* eig_imag)
    {
        T* Ht = host_allocate<T>(Nl*Nl);
        device_2_host_cpy<T>(Ht, (T*)H_device, Nl*Nl);
        hessinberg_eigs(Ht, Nl, eig_real, eig_imag);
        host_deallocate<T>(Ht);
    }



private:
    T* worker = NULL;
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

};



 //LAPACK wrap functions specialization:

template<> inline
void lapack_wrap<double>::hessinberg_eigs(double* H, size_t Nl, double* eig_real, double* eig_imag)
{

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

    dhseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, H,
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
void lapack_wrap<float>::hessinberg_eigs(float* H, size_t Nl, float* eig_real, float* eig_imag)
{

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

    shseqr_(&JOB, &COMPZ, 
            &N, &ILO, 
            &IHI, H,
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





#endif