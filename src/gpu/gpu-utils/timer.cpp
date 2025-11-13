#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include "runtime.hpp"
#include "timer.hpp"

namespace lahva
{
    void GPUTimer::push(std::string label)
    {
        int it;
        it = find(label);

        if (it == -1)
        {
            record.push_back(time_record(label));
            n++;
            it = n - 1;
        }

        last = record[it].label;
        get_cuda_error( cudaEventCreate(&record[it].startEvent));
        get_cuda_error( cudaEventCreate(&record[it].stopEvent));
        if (stream_)
        {
            get_cuda_error(cudaEventRecord(record[it].startEvent, *(stream_)));
        }
        else
        {
            get_cuda_error(cudaEventRecord(record[it].startEvent, 0));
        }
        record[it].running = !record[it].running;
    }

    void GPUTimer::push(std::string label, const cudaStream_t& stream)
    {
        int it;
        it = find(label);

        if (it == -1)
        {
            record.push_back(time_record(label));
            n++;
            it = n - 1;
        }

        last = record[it].label;
        get_cuda_error( cudaEventCreate(&record[it].startEvent));
        get_cuda_error( cudaEventCreate(&record[it].stopEvent));
        get_cuda_error(cudaEventRecord(record[it].startEvent, stream));
        
        record[it].running = !record[it].running;
    }

    void GPUTimer::pop()
    {
        float time;
        int it;

        it = find(last);
        if (it == -1)
            return;

        if (stream_)
        {
            get_cuda_error( cudaEventRecord(record[it].stopEvent, *(stream_)));
        }
        else
        {
            get_cuda_error(cudaEventRecord(record[it].stopEvent, 0));
        }
        get_cuda_error(cudaEventSynchronize(record[it].stopEvent));
        get_cuda_error(cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent));
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty())
            last.clear();
    }

    void GPUTimer::pop(const cudaStream_t& stream)
    {
        float time;
        int it;

        it = find(last);
        if (it == -1)
            return;

        get_cuda_error(cudaEventRecord(record[it].stopEvent, stream));
        
        get_cuda_error(cudaEventSynchronize(record[it].stopEvent));
        get_cuda_error(cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent));
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty())
            last.clear();
    }

    float GPUTimer::get(std::string label)
    {
        float time = -1.0;
        int it;

        if (n <= 0)
            return time;
        it = find(label);
        if (it == -1)
            return time;

        if (record[it].running)
        {
            if (stream_)
            {
                get_cuda_error( cudaEventRecord(record[it].stopEvent, *(stream_)));
            }
            else
            {
                get_cuda_error( cudaEventRecord(record[it].stopEvent, 0));
            };
            get_cuda_error( cudaEventSynchronize(record[it].stopEvent));
            get_cuda_error( cudaEventElapsedTime(&time, record[it].startEvent, record[it].stopEvent));
            time += record[it].time;
        }
        else
            time = record[it].time;

        return time;
    }

}