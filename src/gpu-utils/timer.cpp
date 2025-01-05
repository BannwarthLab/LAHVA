#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include "runtime.hpp"
#include "timer.hpp"


namespace lahva{
    void GPUTimer::push(std::string label) {
        int it;
        cudaError_t istat;
        it = find(label);

        if (it == -1) {
            record.push_back(time_record(label));
            n++;
            it = n-1;
        }

        last = record[it].label;
        istat = cudaEventCreate(&record[it].startEvent);
        istat = cudaEventCreate(&record[it].stopEvent);
        istat = cudaEventRecord(record[it].startEvent, 0);
        get_cuda_error(istat);
        record[it].running = !record[it].running;
    }

    void GPUTimer::pop() {
        float time;
        int it;
        cudaError_t istat;
        
        it = find(last);
        if (it == -1) return;

        istat = cudaEventRecord(record[it].stopEvent, cudart_->getStream());
        istat = cudaEventSynchronize(record[it].stopEvent);
        istat = cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent);
        get_cuda_error(istat);
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty()) last.clear();
    }

    float GPUTimer::get(std::string label) {
        float time = -1.0;
        int it;
        cudaError_t istat;
        
        if (n <= 0) return time;
        it = find(label);
        if (it == -1) return time;

        if (record[it].running) {
            istat = cudaEventRecord(record[it].stopEvent, 0);
            istat = cudaEventSynchronize(record[it].stopEvent);
            istat = cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent);
            get_cuda_error(istat);
            time += record[it].time;
        }
        else
        time = record[it].time;
        
        return time;
    }

}