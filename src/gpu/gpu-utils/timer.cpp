/// @file timer.cpp
/// @brief GPU timer implementation for CUDA event-based profiling.
///
/// Provides GPU-based timing functionality using CUDA events for accurate
/// measurement of GPU kernel and operation execution times.

#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include "runtime.hpp"
#include "timer.hpp"

namespace lahva
{
    /// @brief Starts timing a named operation on the GPU.
    ///
    /// Creates CUDA events and records the start time for a named operation.
    /// If a label already exists, resumes timing; otherwise creates a new timing record.
    ///
    /// @param label Unique identifier for the timed operation.
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

    /// @brief Starts timing a named operation on a specific CUDA stream.
    ///
    /// Creates CUDA events and records the start time for a named operation on the specified stream.
    /// If a label already exists, resumes timing; otherwise creates a new timing record.
    ///
    /// @param label Unique identifier for the timed operation.
    /// @param stream CUDA stream on which to record timing events.
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

    /// @brief Stops timing the most recently started operation.
    ///
    /// Records the stop time for the most recent operation started with push()
    /// and accumulates the elapsed time. Blocks until GPU operations are complete.
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

    /// @brief Stops timing the most recently started operation on a specific CUDA stream.
    ///
    /// Records the stop time for the most recent operation started with push(stream)
    /// and accumulates the elapsed time on the specified stream. Blocks until GPU operations are complete.
    ///
    /// @param stream CUDA stream on which to record timing events.
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

    /// @brief Retrieves accumulated elapsed time for a named operation.
    ///
    /// Returns the total elapsed time in milliseconds for a named operation.
    /// If the operation is currently running, includes time from start to current moment.
    ///
    /// @param label Identifier for the timed operation.
    /// @return Elapsed time in milliseconds, or -1.0 if label not found.
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