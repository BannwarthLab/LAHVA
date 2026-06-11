#include "linalg.hpp"
#include "impl/blas/cpu/additional-level3.hpp"
#include "../../utils/utils.hpp"

namespace lahva
{
    namespace cpu
    {
#ifdef W_MKL
        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const BlockDiagMatrix<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<double> alpha_array, beta_array;
            std::vector<MKL_INT> n_array, ldb_array, ldc_array, m_array, k_array, lda_array, group_size;
            std::vector<const double*> a_array, b_array;
            std::vector<double*> c_array;

            size_t i = 0;
            while (i < a.num_blocks()) {
                MKL_INT current_m = a.get_block(i).shape().first;
                MKL_INT current_k = a.get_block(i).shape().second;
                size_t group_start = i;

                while (i < a.num_blocks() &&
                       a.get_block(i).shape().first == current_m &&
                       a.get_block(i).shape().second == current_k) {
                    ++i;
                }

                size_t group_count = i - group_start;

                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                n_array.push_back(n);
                ldb_array.push_back(k);
                ldc_array.push_back(m);
                m_array.push_back(current_m);
                k_array.push_back(current_k);
                lda_array.push_back(current_m);
                group_size.push_back(group_count);

                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.get_block(j).data());
                    b_array.push_back(b.data() + a.get_col_offsets()[j]);
                    c_array.push_back(c.data() + a.get_row_offsets()[j]);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_dgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const BlockDiagMatrix<double>& b,
                             const double beta, Matrix_<double>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<double> alpha_array, beta_array;
            std::vector<MKL_INT> m_array, n_array, k_array, lda_array, ldb_array, ldc_array, group_size;
            std::vector<const double*> a_array, b_array;
            std::vector<double*> c_array;

            size_t i = 0;
            while (i < b.num_blocks()) {
                MKL_INT current_k = b.get_block(i).shape().first;
                MKL_INT current_n = b.get_block(i).shape().second;
                size_t group_start = i;

                while (i < b.num_blocks() &&
                       b.get_block(i).shape().first == current_k &&
                       b.get_block(i).shape().second == current_n) {
                    ++i;
                }

                size_t group_count = i - group_start;

                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                m_array.push_back(a.shape().first);
                n_array.push_back(current_n);
                k_array.push_back(current_k);
                lda_array.push_back(a.shape().first);
                ldb_array.push_back(current_k);
                ldc_array.push_back(c.shape().first);
                group_size.push_back(group_count);

                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.data() + b.get_col_offsets()[j] * a.shape().first);
                    b_array.push_back(b.get_block(j).data());
                    c_array.push_back(c.data() + b.get_col_offsets()[j] * c.shape().first);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_dgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const BlockDiagMatrix<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<float> alpha_array, beta_array;
            std::vector<MKL_INT> m_array, n_array, k_array, lda_array, ldb_array, ldc_array, group_size;
            std::vector<const float*> a_array, b_array;
            std::vector<float*> c_array;

            size_t i = 0;
            while (i < a.num_blocks()) {
                MKL_INT current_m = a.get_block(i).shape().first;
                MKL_INT current_k = a.get_block(i).shape().second;
                size_t group_start = i;

                while (i < a.num_blocks() &&
                       a.get_block(i).shape().first == current_m &&
                       a.get_block(i).shape().second == current_k) {
                    ++i;
                }

                size_t group_count = i - group_start;

                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                m_array.push_back(current_m);
                n_array.push_back(n);
                k_array.push_back(current_k);
                lda_array.push_back(current_m);
                ldb_array.push_back(k);
                ldc_array.push_back(m);
                group_size.push_back(group_count);

                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.get_block(j).data());
                    b_array.push_back(b.data() + a.get_col_offsets()[j]);
                    c_array.push_back(c.data() + a.get_row_offsets()[j]);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_sgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const BlockDiagMatrix<float>& b,
                             const float beta, Matrix_<float>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<float> alpha_array, beta_array;
            std::vector<MKL_INT> m_array, n_array, k_array, lda_array, ldb_array, ldc_array, group_size;
            std::vector<const float*> a_array, b_array;
            std::vector<float*> c_array;

            size_t i = 0;
            while (i < b.num_blocks()) {
                MKL_INT current_k = b.get_block(i).shape().first;
                MKL_INT current_n = b.get_block(i).shape().second;
                size_t group_start = i;

                while (i < b.num_blocks() &&
                       b.get_block(i).shape().first == current_k &&
                       b.get_block(i).shape().second == current_n) {
                    ++i;
                }

                size_t group_count = i - group_start;

                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                m_array.push_back(a.shape().first);
                n_array.push_back(current_n);
                k_array.push_back(current_k);
                lda_array.push_back(a.shape().first);
                ldb_array.push_back(current_k);
                ldc_array.push_back(c.shape().first);
                group_size.push_back(group_count);

                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.data() + b.get_col_offsets()[j] * a.shape().first);
                    b_array.push_back(b.get_block(j).data());
                    c_array.push_back(c.data() + b.get_col_offsets()[j] * c.shape().first);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_sgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }



#else

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const BlockDiagMatrix<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c)
        {
            
            int p = b.shape().second;

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
                int m = a.block_shapes()[i][0];
                int n = a.block_shapes()[i][1];
            
                const double* Ai = static_cast<const double*>(a.get_block_data(i));

                // Compute pointers based on pre-computed stride
                const double* Bi = b.data() + a.get_col_offsets()[i];
                double* Ci = c.data() + a.get_row_offsets()[i];

                cblas_dgemm(
                    major,
                    transa,
                    transb,
                    m, p, n,
                    alpha,
                    Ai, m,
                    Bi, ldb,
                    beta,
                    Ci, ldc
        );
    }
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const BlockDiagMatrix<float>& a, const Matrix_<float>& b,
                             const double beta, Matrix_<float>& c)
        {
            
            int p = b.shape().second;

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
                int m = a.block_shapes()[i][0];
                int n = a.block_shapes()[i][1];
            
                const float* Ai = static_cast<const float*>(a.get_block_data(i));

                // Compute pointers based on pre-computed stride
                const float* Bi = b.data() + a.get_col_offsets()[i];
                float* Ci = c.data() + a.get_row_offsets()[i];

                cblas_sgemm(
                    major,
                    transa,
                    transb,
                    m, p, n,
                    alpha,
                    Ai, m,
                    Bi, ldb,
                    beta,
                    Ci, ldc
        );
    }
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const BlockDiagMatrix<double>& b,
                             const double beta, Matrix_<double>& c)
        {
            
            int m = a.shape().first;

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            
            int rows, cols, inner;
            std::tie(rows, cols, inner) = check_size_mm(a, b, c, transa, transb);
            BLAS_INT lda = get_leading(rows, inner);
            BLAS_INT ldc = get_leading(rows, cols);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < b.num_blocks(); ++i) {
                int b_rows = b.get_block(i).shape().first;
                int b_cols = b.get_block(i).shape().second;
            
                const double* Ai = a.data() + b.get_col_offsets()[i] * m;
                const double* Bi = b.get_block(i).data();
                double* Ci = c.data() + b.get_col_offsets()[i] * m;

                cblas_dgemm(
                    major,
                    transa,
                    transb,
                    m, b_cols, b_rows,
                    alpha,
                    Ai, lda,
                    Bi, b_rows,
                    beta,
                    Ci, ldc
        );
    }
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const BlockDiagMatrix<float>& b,
                             const float beta, Matrix_<float>& c)
        {
            
            int m = a.shape().first;

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            
            int rows, cols, inner;
            std::tie(rows, cols, inner) = check_size_mm(a, b, c, transa, transb);
            BLAS_INT lda = get_leading(rows, inner);
            BLAS_INT ldc = get_leading(rows, cols);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < b.num_blocks(); ++i) {
                int b_rows = b.get_block(i).shape().first;
                int b_cols = b.get_block(i).shape().second;
            
                const float* Ai = a.data() + b.get_col_offsets()[i] * m;
                const float* Bi = b.get_block(i).data();
                float* Ci = c.data() + b.get_col_offsets()[i] * m;

                cblas_sgemm(
                    major,
                    transa,
                    transb,
                    m, b_cols, b_rows,
                    alpha,
                    Ai, lda,
                    Bi, b_rows,
                    beta,
                    Ci, ldc
        );
    }
        }

#endif

}  // namespace cpu
}  // namespace lahva