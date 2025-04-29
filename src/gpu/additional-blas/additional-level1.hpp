#pragma once
namespace lahva
{
    namespace gpu
    {
        template<typename in, typename out>
        void CopyTensors(const unsigned long size, const in* d_in, out* d_out);

        
    } // namespace gpu
    
} // namespace lahva
