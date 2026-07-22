#pragma once
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU device functor for computing absolute value (fabs operation).
        ///
        ///
        /// @tparam T Numerical type
        template<typename T>
        class fabs_gpu
        {
        public:
            /// @brief Compute absolute value on GPU device.
            ///
            /// @param x Input value.
            /// @return Absolute value of x (|x|).
            __device__ T operator()(T x)
            {
                return fabs(x);
            }
        };

    } // namespace gpu

} // namespace lahva
