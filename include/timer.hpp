#ifndef TCGMBLAS_GPUTIMER_HPP
#define TCGMBLAS_GPUTIMER_HPP
#include "runtime.hpp"
#include <chrono>
#include <string>
#include <vector>

namespace tcgmtensor{
class time_record {
    public:
        std::string label;
        bool running = false;
        float time = 0.0;
        cudaEvent_t startEvent = 0, stopEvent = 0;
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime, stopTime;
        time_record() {};
        time_record(std::string lbl) : label{lbl}{};
        ~time_record() {};
};


class Timer {
protected:
    size_t n = 0;
    std::string last = "";
    std::vector<time_record> record;    
public:
    Timer() {};
    virtual ~Timer() {};
    public:
        void virtual push(std::string label) = 0;
        void virtual pop() = 0 ;
        float virtual get(std::string label) = 0;
        void print_entries();
    protected:
        int find(std::string label);
        void resize(int n);
};

class GPUTimer : public Timer{

public:
    GPUTimer() {};
    ~GPUTimer() {};
    public:
        void push(std::string label) override;
        void pop() override;
        float get(std::string label) override;
};

class CPUTimer : public Timer{
public:
    CPUTimer() {};
    ~CPUTimer() {};
    public:
        void push(std::string label) override;
        void pop() override;
        float get(std::string label) override;
};


std::string format_time(float time);
std::string format_time(double time);
}



#endif