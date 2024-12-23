#include "runtime.hpp"
#include "linalg.hpp"
#include "tcgmblas.hpp"
#include "timer.hpp"
#include <iostream>
#include <omp.h>
#include <vector>
using namespace tcgmtensor;
template<typename T>
using CPUAll = CudaHostAllocator<T>;
template<typename T>
using GPUAll = CudaDeviceAsyncAllocator<T>;

template<typename T>
using Vector = typename gpu::Vector<T,CPUAll<T>,GPUAll<T>>;
template<typename T>
using Matrix = typename gpu::Matrix<T,CPUAll<T>,GPUAll<T>>;
template<typename T>
using LowTriMatrix = typename gpu::LowTriMatrix<T,CPUAll<T>,GPUAll<T>>;

class dummy
{
    protected:
        CudaRuntime cudart = CudaRuntime(true);
        Matrix<float> v;
        Vector<double> vp;
        Matrix<float> v1;
        Vector<double> vp1;
        Matrix<float> vres;
        Vector<double> vpres;
    public:
        dummy() {};
        dummy(int n, bool async = true) : v(Shape(n,n),1.0), v1(Shape(n,n),1.0), vres(Shape(n,n),1.0), vp(n, 1.0), vp1(n,1.0), vpres(n, 0.0) 
        {   
            if (async) cudart.createStream();
            v.allocateGPU(cudart);
            vp.allocateGPU(cudart);
            v1.allocateGPU(cudart);
            vp1.allocateGPU(cudart);
            vres.allocateGPU(cudart);
            vpres.allocateGPU(cudart);
        };
        ~dummy() 
        {
            v.deallocateGPU(cudart);
            vp.deallocateGPU(cudart);
            v1.deallocateGPU(cudart);
            vp1.deallocateGPU(cudart);
            vres.deallocateGPU(cudart);
            vpres.deallocateGPU(cudart);
        };



        void matmul()
        {
            //gpu::MatrixMatrixProduct(cudart, v, v1, vres);
            gpu::MPSymMatrixMatrixMultiplication(cudart, vp, v, vp1, v1, vpres, vres);
            gpu::ComputeTrace(cudart, v);
        };
        void print() 
        {
            vres.copy2host(cudart);
            cudart.synchronize();
            std::cout << vres.sum() << std::endl;
        };
        void sync()
        {
            cudart.synchronize();
        };

        dummy& operator=(const dummy& other)
        {
            if (this != &other)
            {
            this->cudart = other.cudart;
            this->v = other.v;
            this->vp = other.vp;
            this->v1 = other.v1;
            this->vp1 = other.vp1;
            this->vres = other.vres;
            this->vpres = other.vpres;
            }
            return *this;
        }
        

};

class dummyd
{
    protected:
        CudaRuntime cudart = CudaRuntime(false);
        Matrix<float> v;
        Vector<double> vp;
        Matrix<float> v1;
        Vector<double> vp1;
        Matrix<float> vres;
        Vector<double> vpres;
    public:
        dummyd(int n, bool async = true) : v(Shape(n,n),1.0), v1(Shape(n,n),1.0), vres(Shape(n,n),1.0), vp(n, 1.0), vp1(n,1.0), vpres(n, 0.0) 
        {
            if (async) cudart.createStream();
        };
        ~dummyd() {};



        void matmul()
        {
            gpu::MatrixMatrixProduct(cudart, v, v1, vres);
            gpu::MPSymMatrixMatrixMultiplication(cudart, vp, v, vp1, v1, vpres, vres);
            gpu::ComputeTrace(cudart, v);
        };
        void print() 
        {
            vres.copy2host(cudart);
            cudart.synchronize();
            std::cout << vres.sum() << std::endl;
        };
        void sync()
        {
            cudart.synchronize();
        };

        

};

void parallel(int n, int threads, CPUTimer& timer)
{
    //omp_set_num_threads(4);
    

    timer.push("OMP parallel");
    #pragma omp parallel
    #pragma omp single
    {
    for (int i =0; i<threads ; i++)
    {
        
        omp_set_num_threads(2);
        #pragma omp task firstprivate(i)
        {
            int thread_id = i ;
            std::cout << thread_id << std::endl;
            dummy vecs(n, true);
            vecs.matmul();
            vecs.matmul();
            vecs.matmul();
            //vec[thread_id].sync();
        }   
    }
    }
    timer.pop();
}

void serial(int n, int threads, CPUTimer& timer)
{


    timer.push("serial detach, async");
    for (int i =0; i<threads ; i++)
    {
        
        int thread_id = i;
        std::cout << thread_id << std::endl;
        dummy vecs(n, false);
        vecs.matmul();
        vecs.matmul();
        vecs.matmul();
        
    }
    timer.pop();
}

void serial_stupid(int n, int threads, CPUTimer& timer)
{
  

    timer.push("serial detach, ");
    for (int i =0; i<threads ; i++)
    {
        dummyd vec(n, false);
        int thread_id = i;
        std::cout << thread_id << std::endl;
            
        vec.matmul();
        vec.matmul();
        vec.matmul();
        
    }
    timer.pop();
}


int main()
{
    int n = 8000;
    int threads = 30;

   
    CPUTimer timer;

    serial(n ,threads, timer);

    parallel(n, threads, timer);    

    serial_stupid(n, threads, timer);


    timer.print_entries();
}