/// @file print.hpp
/// @brief Shared formatting helpers for printing tensors.
///
/// Every tensor print() routes through write_table(), which right-aligns each
/// column, formats entries with a given number of significant digits, prints
/// entries below a zero tolerance as 0, and can emit a LaTeX tabular instead of
/// plain text.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace lahva
{
    /// @brief Output format for tensor print() functions
    enum class PrintFormat
    {
        plain, ///< aligned columns separated by spaces
        latex  ///< LaTeX tabular environment
    };

    namespace print_detail
    {
        /// @brief Trait that is true for std::complex<U>
        template <typename T>
        struct is_complex : std::false_type
        {
        };

        /// @brief Trait that is true for std::complex<U>
        template <typename U>
        struct is_complex<std::complex<U>> : std::true_type
        {
        };

        /// @brief Convert a real element (including __half) to double
        /// @tparam T element type
        /// @param[in] v element value
        /// @return v as double
        template <typename T>
        inline double to_double(const T &v)
        {
            if constexpr (std::is_arithmetic_v<T>)
                return static_cast<double>(v);
            else
                // e.g. __half: has several conversion operators, float is unambiguous
                return static_cast<double>(static_cast<float>(v));
        }

        /// @brief Format a real number with `digits` significant digits, or as 0 below `zero_tol`
        /// @param[in] x value
        /// @param[in] digits number of significant digits
        /// @param[in] zero_tol values with |x| < zero_tol are printed as 0
        /// @return formatted value
        inline std::string format_real(double x, int digits, double zero_tol)
        {
            if (std::abs(x) < zero_tol)
                return "0";
            std::ostringstream os;
            os << std::setprecision(digits) << x;
            return os.str();
        }

        /// @brief Format one tensor element
        /// @tparam T element type
        /// @param[in] v element value
        /// @param[in] digits number of significant digits (ignored for integers)
        /// @param[in] zero_tol entries with |v| < zero_tol are printed as 0
        /// @return formatted element
        template <typename T>
        std::string format_value(const T &v, int digits, double zero_tol)
        {
            if constexpr (is_complex<T>::value)
            {
                if (std::abs(v) < zero_tol)
                    return "0";
                const double re = static_cast<double>(v.real());
                const double im = static_cast<double>(v.imag());
                std::string out = format_real(re, digits, zero_tol);
                if (std::abs(im) < zero_tol)
                    return out;
                out += (im < 0 ? "-" : "+");
                out += format_real(std::abs(im), digits, 0.0);
                out += "i";
                return out;
            }
            else if constexpr (std::is_integral_v<T>)
            {
                // widen so int8_t/uint8_t print as numbers instead of characters
                return std::to_string(static_cast<long long>(v));
            }
            else
            {
                return format_real(to_double(v), digits, zero_tol);
            }
        }

        /// @brief Write rows of formatted cells as a right-aligned table
        /// @param[in,out] os output stream
        /// @param[in] rows table cells; rows may be shorter than the widest row
        /// @param[in] format plain text or LaTeX tabular
        /// @param[in] n_label_cols number of leading label columns (e.g. an index column);
        ///            each gets its own width, all other cells share one common width
        inline void write_table(std::ostream &os, const std::vector<std::vector<std::string>> &rows, PrintFormat format,
                                size_t n_label_cols = 0)
        {
            size_t n_cols = 0;
            for (const auto &row : rows)
                n_cols = std::max(n_cols, row.size());

            // label columns: own width; element columns: one width for the whole tensor
            std::vector<size_t> width(n_cols, 0);
            size_t element_width = 0;
            for (const auto &row : rows)
                for (size_t j = 0; j < row.size(); j++)
                {
                    if (j < n_label_cols)
                        width[j] = std::max(width[j], row[j].size());
                    else
                        element_width = std::max(element_width, row[j].size());
                }
            for (size_t j = n_label_cols; j < n_cols; j++)
                width[j] = element_width;

            const bool latex = (format == PrintFormat::latex);
            if (latex)
                os << "\\begin{tabular}{" << std::string(n_cols, 'r') << "}\n";

            for (const auto &row : rows)
            {
                // plain rows stop at their last cell; latex rows need every column
                const size_t n = latex ? n_cols : row.size();
                for (size_t j = 0; j < n; j++)
                {
                    if (j > 0)
                        os << (latex ? " & " : "  ");
                    os << std::setw(static_cast<int>(width[j])) << (j < row.size() ? row[j] : std::string());
                }
                if (latex)
                    os << " \\\\";
                os << '\n';
            }

            if (latex)
                os << "\\end{tabular}\n";
            os.flush();
        }

        /// @brief Open a file (truncating it) and pass the stream to a writer
        /// @tparam Write callable (std::ostream &) -> void
        /// @param[in] file output filename
        /// @param[in] write function that writes the content
        /// @note Prints an error to stderr and writes nothing if the file cannot be opened
        template <typename Write>
        void write_file(const char *file, Write write)
        {
            std::ofstream os(file);
            if (!os.is_open())
            {
                std::cerr << "Unable to open file " << file << std::endl;
                return;
            }
            write(os);
        }

        /// @brief Format a matrix through an element accessor and write it as a table
        /// @tparam T element type
        /// @tparam Get callable (i, j) -> T
        /// @param[in,out] os output stream
        /// @param[in] n_rows number of rows
        /// @param[in] n_cols number of columns
        /// @param[in] get element accessor
        /// @param[in] digits number of significant digits
        /// @param[in] zero_tol entries with |a_ij| < zero_tol are printed as 0
        /// @param[in] format plain text or LaTeX tabular
        /// @param[in] lower_only only print entries with j <= i (lower triangle)
        template <typename T, typename Get>
        void write_matrix(std::ostream &os, size_t n_rows, size_t n_cols, Get get,
                          int digits, double zero_tol, PrintFormat format, bool lower_only = false)
        {
            std::vector<std::vector<std::string>> rows(n_rows);
            for (size_t i = 0; i < n_rows; i++)
            {
                const size_t n = lower_only ? std::min(i + 1, n_cols) : n_cols;
                rows[i].reserve(n);
                for (size_t j = 0; j < n; j++)
                    rows[i].push_back(format_value<T>(get(i, j), digits, zero_tol));
            }
            write_table(os, rows, format);
        }

        /// @brief Format a vector as a column, optionally preceded by the element index
        /// @tparam T element type
        /// @param[in,out] os output stream
        /// @param[in] data pointer to the elements
        /// @param[in] n number of elements
        /// @param[in] digits number of significant digits
        /// @param[in] zero_tol entries with |v_i| < zero_tol are printed as 0
        /// @param[in] format plain text or LaTeX tabular
        /// @param[in] show_index print the (0-based) index before each element
        template <typename T>
        void write_vector(std::ostream &os, const T *data, size_t n,
                          int digits, double zero_tol, PrintFormat format, bool show_index)
        {
            std::vector<std::vector<std::string>> rows(n);
            for (size_t i = 0; i < n; i++)
            {
                if (show_index)
                    rows[i].push_back(std::to_string(i));
                rows[i].push_back(format_value<T>(data[i], digits, zero_tol));
            }
            write_table(os, rows, format, show_index ? 1 : 0);
        }
    } // namespace print_detail

    /// @brief Default zero tolerance for print(): machine epsilon of T
    /// @tparam T element type
    /// @return epsilon of T (of its value type for complex, 2^-10 for __half, 0 for integers)
    template <typename T>
    constexpr double default_zero_tol()
    {
        if constexpr (print_detail::is_complex<T>::value)
            return std::numeric_limits<typename T::value_type>::epsilon();
        else if constexpr (std::is_integral_v<T>)
            return 0.0;
        else if constexpr (std::is_floating_point_v<T>)
            return std::numeric_limits<T>::epsilon();
        else
            return 9.765625e-4; // __half: 2^-10
    }
} // namespace lahva
