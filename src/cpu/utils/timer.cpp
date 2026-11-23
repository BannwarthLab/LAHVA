/// @file timer.cpp
/// @brief CPU timing and performance measurement utilities implementation.
///
/// Provides CPUTimer class implementation for tracking and recording high-resolution
/// timing measurements of CPU operations, useful for performance profiling and benchmarking.

#include <iostream>
#include <sstream>
#include "timer.hpp"

namespace lahva
{

    /// @brief Start timing a labeled operation.
    /// @param label Unique identifier for the timed operation.
    void CPUTimer::push(std::string label) {
        int it;
        it = find(label);

        if (it == -1) {
            record.push_back(time_record(label));
            n++;
            it = n-1;
        }

        last = record[it].label;
        record[it].startTime = std::chrono::high_resolution_clock::now();
        record[it].running = !record[it].running;
    }

    /// @brief Stop timing the currently active operation.
    void CPUTimer::pop() {
        int it;
        float time;
        
        it = find(last);
        if (it == -1) return;

        record[it].stopTime = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<float, std::milli>(record[it].stopTime - record[it].startTime).count();
        record[it].time += time;
        record[it].running = !record[it].running;
        if (!last.empty()) last.clear();
    }

    /// @brief Retrieve accumulated time for a labeled operation.
    /// @param label Unique identifier of the operation.
    /// @return Accumulated time in milliseconds, or -1.0 if label not found.
    float CPUTimer::get(std::string label) {
        float time = -1.0;
        int it;
        
        if (n <= 0) return time;
        it = find(label);
        if (it == -1) return time;

        if (record[it].running) {
            record[it].stopTime = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration<float, std::milli>(record[it].stopTime - record[it].startTime).count();
            time += record[it].time;
        }
        else
        time = record[it].time;
        
        return time;
    }

    /// @brief Find the index of a timer record by label.
    /// @param label Unique identifier to search for.
    /// @return Index of the record if found, -1 otherwise.
    int Timer::find(std::string label) {
        int pos = -1;

        for (size_t i = 0; i < record.size(); i++) {
            if (record[i].label == label) {
                pos = i;
                break;
            }
        }
        return pos;
    }

    /// @brief Resize the timer records vector.
    /// @param n New size for the records collection.
    void Timer::resize(int n) {
        record.resize(n);
    }

    /// @brief Generate a formatted string representation of all timing records.
    /// @return Formatted string with all timer entries and total time.
    std::string Timer::print_entries()
    {
        std::ostringstream oss;
        oss << "Timings" << std::endl;
        float ttime = 0.0; 
        for (time_record entry : record) 
        {
            float time = this->get(entry.label);
            oss << " - " << entry.label << " :\t \t" << format_time(time) << std::endl;
            ttime += time;
        }

        oss << "-----------------------------------------------------" << std::endl;
        oss << "Total: \t\t\t" << format_time(ttime) <<std::endl;

        return oss.str();

    }

    /// @brief Format milliseconds into a human-readable time string.
    /// @param time Time in milliseconds.
    /// @return Formatted string (e.g., "2 d, 3 h, 15 min, 30.5 sec").
    std::string format_time(double time) {
        int days, hours, mins;
        double secs;
    std::string str;

    time = time / 1000; 

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

    /// @brief Format milliseconds into a human-readable time string (float overload).
    /// @param time Time in milliseconds.
    /// @return Formatted string (e.g., "2 d, 3 h, 15 min, 30.5 sec").
    std::string format_time(float time) {
        return format_time((double)time);
    }

} // namespace lahva