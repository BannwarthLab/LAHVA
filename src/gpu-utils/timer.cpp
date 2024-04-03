#include <cstddef>
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>



class time_record {
public:
    std::string label;
    bool running = false;
    float time = 0.0;
    cudaEvent_t startEvent = 0, stopEvent = 0;
};

class gputimer_type {
public:
    size_t n = 0;
    std::string last;
    std::vector<time_record> record;

    void push(std::string label) {
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
        record[it].running = !record[it].running;
    }

    void pop() {
        float time;
        int it, istat;

        it = find(last);
        if (it == -1) return;

        istat = cudaEventRecord(record[it].stopEvent, 0);
        istat = cudaEventSynchronize(record[it].stopEvent);
        istat = cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent);
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty()) last.clear();
    }

    float get(std::string label) {
        float time = 0.0;
        int it, istat;

        if (n <= 0) return time;
        it = find(label);
        if (it == -1) return time;

        if (record[it].running) {
            istat = cudaEventRecord(record[it].stopEvent, 0);
            istat = cudaEventSynchronize(record[it].stopEvent);
            istat = cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent);
            time += record[it].time;
        }
        return time;
    }

    int find(std::string label) {
        int pos = -1;

        for (int i = record.size() - 1; i >= 0; i--) {
            if (record[i].label == label) {
                pos = i;
                break;
            }
        }
        return pos;
    }

    void resize(int n) {
        record.resize(n);
    }
};

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

int main() {
    // Example usage of the gputimer_type class
    gputimer_type timer;

    timer.push("Test1");
    // Run some CUDA code
    timer.pop();

    timer.push("Test2");
    // Run some more CUDA code
    timer.pop();

    std::cout << "Time for Test1: " << format_time_dp(timer.get("Test1")) << std::endl;
    std::cout << "Time for Test2: " << format_time_dp(timer.get("Test2")) << std::endl;

    return 0;
}
