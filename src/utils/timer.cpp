#include "timer.hpp"
#include <string>
#include <iostream>
#include <chrono>

namespace lahva{
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

    void Timer::resize(int n) {
        record.resize(n);
    }

    void Timer::print_entries()
    {
        float ttime = 0.0; 
        for (time_record entry : record) 
        {
            float time = this->get(entry.label);
            std::cout << " - " << entry.label << " :\t \t" << format_time(time) << std::endl;
            ttime += time;
        }

        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << "Total: \t\t" << format_time(ttime) <<std::endl;
    }


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

std::string format_time(float time) {
    return format_time((double)time);
}
}