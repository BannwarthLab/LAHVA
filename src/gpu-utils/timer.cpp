#include <cstddef>
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "runtime.hpp"
#include "timer.hpp"


namespace tcgmtensor{
    void GPUTimer::push(std::string label) {
        int it;
        cudaError_t istat;

        it = find(label);

        if (it == -1) {
            if (n >= record.size()) {
                resize(record.size() * 2 + 1);
            }

            n++;
            it = n;
            record[it].label = label;
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

        istat = cudaEventRecord(record[it].stopEvent, 0);
        istat = cudaEventSynchronize(record[it].stopEvent);
        istat = cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent);
        get_cuda_error(istat);
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty()) last.clear();
    }

    float GPUTimer::get(std::string label) {
        float time = 0.0;
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
        return time;
    }

    int GPUTimer::find(std::string label) {
        int pos = -1;

        for (int i = record.size() - 1; i >= 0; i--) {
            if (record[i].label == label) {
                pos = i;
                break;
            }
        }
        return pos;
    }

    void GPUTimer::resize(int n) {
        record.resize(n);
    }


std::string format_time_dp(double time) {
    int days, hours, mins;
    double secs;
    std::string str;

    days = static_cast<int>(time / 86400.0);
    time -= days * 86400.0;
    hours = static_cast<int>(time / 3600.0);
    time -= hours * 3600.0;
    mins = static_cast<int>(time / 60.0);
    time -= mins * 60.0;
    secs = time;

    if (days > 0) str += std::to_string(days) + " d, ";
    if (hours > 0) str += std::to_string(hours) + " h, ";
    if (mins > 0) str += std::to_string(mins) + " min, ";
    str += std::to_string(secs) + " sec";

    return str;
}

std::string format_time_sp(float time) {
    return format_time_dp(time);
}

}