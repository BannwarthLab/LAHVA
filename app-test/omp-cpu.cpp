#include "const.h"
#include "linalg.hpp"
#include "tcgmblas.hpp"
#include "timer.hpp"
#include <iostream>
#include <omp.h>
#include <vector>
using namespace tcgmtensor;
template<typename T>
using Matrix = cpu::Matrix<T>;
template<typename T>
using Vector = cpu::Vector<T>;

class dummy
{
    protected:
        CPURuntime cudart = CPURuntime();
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
            
        };
        ~dummy() 
        {
            
        };



        void matmul()
        {
            cpu::MatrixMatrixProduct(cudart, v, v1, vres);
            //cpu::MPSymMatrixMatrixMultiplication(cudart, vp, v, vp1, v1, vpres, vres);
            cpu::ComputeTrace(cudart, v);
        };
        void print() 
        {
            
            std::cout << vres.sum() << std::endl;
        };
        void sync()
        {
            
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
        CPURuntime cudart = CPURuntime();
        Matrix<float> v;
        Vector<double> vp;
        Matrix<float> v1;
        Vector<double> vp1;
        Matrix<float> vres;
        Vector<double> vpres;
    public:
        dummyd(int n, bool async = true) : v(Shape(n,n),1.0), v1(Shape(n,n),1.0), vres(Shape(n,n),1.0), vp(n, 1.0), vp1(n,1.0), vpres(n, 0.0) 
        {
            
        };
        ~dummyd() {};



        void matmul()
        {
            cpu::MatrixMatrixProduct(cudart, v, v1, vres);
            //cpu::MPSymMatrixMatrixMultiplication(cudart, vp, v, vp1, v1, vpres, vres);
            cpu::ComputeTrace(cudart, v);
        };
        void print() 
        {
            
            std::cout << vres.sum() << std::endl;
        };
        void sync()
        {
            
        };

        

};

void parallel(int n, int threads, CPUTimer& timer)
{
    omp_set_num_threads(2);
    

    timer.push("OMP parallel");
    #pragma omp parallel
    #pragma omp single
    {
    for (int i =0; i<threads ; i++)
    {
        

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

    omp_set_num_threads(4);
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
  
    omp_set_num_threads(4);
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

    Vector<int> a(3);
    Vector<double> ab(5,a.get_allocator());
    ab.print();

    Matrix<double> m(Shape(2,2),a.get_allocator());

    serial(n ,threads, timer);

    parallel(n, threads, timer);    

    serial_stupid(n, threads, timer);


    timer.print_entries();
}