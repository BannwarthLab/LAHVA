/// @file timer.hpp
/// @brief Timing utilities for measuring CPU and GPU computation performance.
///
/// This header provides timer implementations for profiling execution time on both
/// CPU and GPU.

#pragma once
#ifdef _CUDA
#include "runtime.hpp"
#else
typedef unsigned long long cudaEvent_t;
#endif
#include <chrono>
#include <string>
#include <vector>

namespace lahva{

/// @brief Record of a single timing measurement
/// Stores timing data for both CPU and GPU measurements with labels.
class time_record {
    public:
        /// @brief Label identifying this timing record
        std::string label;

        /// @brief Flag indicating if timer is currently running
        bool running = false;

        /// @brief Elapsed time in milliseconds
        float time = 0.0;

        /// @brief CUDA event for GPU timing start
        cudaEvent_t startEvent = 0;

        /// @brief CUDA event for GPU timing stop
        cudaEvent_t stopEvent = 0;

        /// @brief CPU clock time point at start
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

        /// @brief CPU clock time point at stop
        std::chrono::time_point<std::chrono::high_resolution_clock> stopTime;

        /// @brief Default constructor
        time_record() {};

        /// @brief Construct time record with label
        /// @param[in] lbl label for this timing record
        time_record(std::string lbl) : label{lbl}{};

        /// @brief Destructor
        ~time_record() {};
};

/// @brief Abstract base timer class for profiling CPU and GPU execution
/// Provides interface for starting/stopping named timers and retrieving elapsed time.
class Timer {
protected:
    /// @brief Number of timing records
    size_t n = 0;

    /// @brief Label of most recently started timer
    std::string last = "";

    /// @brief Vector of timing records
    std::vector<time_record> record;

public:
    /// @brief Default constructor
    Timer() {};

    /// @brief Virtual destructor
    virtual ~Timer() {};

    /// @brief Start a named timer
    /// @param[in] label identifier for this timing measurement
    void virtual push(std::string label) = 0;

    /// @brief Stop the currently running timer
    void virtual pop() = 0;

    /// @brief Get elapsed time for a labeled measurement
    /// @param[in] label identifier for timing record to retrieve
    /// @return elapsed time in milliseconds
    float virtual get(std::string label) = 0;

    /// @brief Format and return all timing records as a string
    /// @return formatted string with all timing entries
    std::string print_entries();

protected:
    /// @brief Find index of timing record by label
    /// @param[in] label label to search for
    /// @return index of record, or -1 if not found
    int find(std::string label);

    /// @brief Resize the timing records vector
    /// @param[in] n new size for records vector
    void resize(int n);
};
#ifdef _CUDA
/// @brief GPU-based timer using CUDA events for precise GPU timing
/// Measures execution time on GPU using CUDA events with optional stream specification.
class GPUTimer : public Timer{
    protected:
        /// @brief CUDA stream pointer for GPU operations
        mutable std::shared_ptr<cudaStream_t> stream_;

    public:
        /// @brief Default constructor for GPU timer
        GPUTimer() {};

        /// @brief Construct GPU timer with CUDA runtime
        /// @param[in] cudart CUDA runtime instance to extract stream pointer from
        GPUTimer(CudaRuntime* cudart) {stream_ = cudart->getStreamPtr();};

        /// @brief Destructor
        ~GPUTimer() {};

    public:
        /// @brief Start named GPU timer on current stream
        /// @param[in] label identifier for this timing measurement
        void push(std::string label) override;

        /// @brief Start named GPU timer on specified stream
        /// @param[in] label identifier for this timing measurement
        /// @param[in] stream CUDA stream to use for timing
        void push(std::string label, const cudaStream_t& stream);

        /// @brief Stop the currently running GPU timer on current stream
        void pop() override;

        /// @brief Stop the currently running GPU timer on specified stream
        /// @param[in] stream CUDA stream to use for timing
        void pop(const cudaStream_t& stream);

        /// @brief Get elapsed time for labeled GPU measurement
        /// @param[in] label identifier for timing record to retrieve
        /// @return elapsed time in milliseconds
        float get(std::string label) override;
};
#endif
/// @brief CPU-based timer using high-resolution system clock
/// Measures execution time on CPU using std::chrono high_resolution_clock.
class CPUTimer : public Timer{
public:
    /// @brief Default constructor for CPU timer
    CPUTimer() {};

    /// @brief Destructor
    ~CPUTimer() {};

    /// @brief Start named CPU timer
    /// @param[in] label identifier for this timing measurement
    void push(std::string label) override;

    /// @brief Stop the currently running CPU timer
    void pop() override;

    /// @brief Get elapsed time for labeled CPU measurement
    /// @param[in] label identifier for timing record to retrieve
    /// @return elapsed time in milliseconds
    float get(std::string label) override;
};

/// @brief Format floating-point time value into human-readable string
/// @param[in] time time in milliseconds as float
/// @return formatted time string with appropriate units (ms, s, etc.)
std::string format_time(float time);

/// @brief Format floating-point time value into human-readable string
/// @param[in] time time in milliseconds as double
/// @return formatted time string with appropriate units (ms, s, etc.)
std::string format_time(double time);
}


