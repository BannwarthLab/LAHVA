#ifndef TCGMBLAS_GPUTIMER_HPP
#define TCGMBLAS_GPUTIMER_HPP
#include "runtime.hpp"

namespace tcgmtensor{
class time_record {
    public:
        std::string label;
        bool running = false;
        float time = 0.0;
        cudaEvent_t startEvent = 0, stopEvent = 0;
};

class GPUTimer {
public:
    size_t n = 0;
    std::string last = "";
    std::vector<time_record> record;

    GPUTimer() {};
    ~GPUTimer() {};
    public:
        void push(std::string label);
        void pop() ;
        float get(std::string label);
    protected:
    int find(std::string label);
    void resize(int n);
};
}

#endif