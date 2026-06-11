#pragma once

namespace lahva
{
    namespace gpu
    {
        //! @brief Enumeration for sparse matrix format selection
        enum class SparseFormat
        {
            CSR,        //!< Compressed Sparse Row format
            BSR         //!< Block Sparse Row format
        };
    } // namespace gpu
} // namespace lahva
