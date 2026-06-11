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
            std::vector<CBLAS_TRANSPOSE> transa_array(a.num_blocks(), transa), transb_array(a.num_blocks(), transb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            
            std::vector<double> alpha_array(a.num_blocks(), alpha);
            std::vector<double> beta_array(a.num_blocks(), beta);
            std::vector<MKL_INT> n_array(a.num_blocks(), n);
            std::vector<MKL_INT> ldb_array(a.num_blocks(), k);
            std::vector<MKL_INT> ldc_array(a.num_blocks(), m);
            std::vector<MKL_INT> group_size(a.num_blocks(), 1);

            std::vector<MKL_INT> m_array(a.num_blocks()), k_array(a.num_blocks()), lda_array(a.num_blocks());
            std::vector<const double*> a_array(a.num_blocks()), b_array(a.num_blocks());
            std::vector<double*> c_array(a.num_blocks());

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
            
                m_array[i] = a.get_block(i).shape().first;
                k_array[i] = a.get_block(i).shape().second;
                lda_array[i] = a.get_block(i).shape().first;

                a_array[i] = a.get_block(i).data();
                b_array[i] = b.data() + a.get_col_offsets()[i];
                c_array[i] = c.data() + a.get_row_offsets()[i];
            }

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
                a.num_blocks(),
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const BlockDiagMatrix<double>& b,
                             const double beta, Matrix_<double>& c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            std::vector<CBLAS_TRANSPOSE> transa_array(b.num_blocks(), transa), transb_array(b.num_blocks(), transb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            
            std::vector<double> alpha_array(b.num_blocks(), alpha);
            std::vector<double> beta_array(b.num_blocks(), beta);
            std::vector<MKL_INT> m_array(b.num_blocks(), a.shape().first);
            std::vector<MKL_INT> lda_array(b.num_blocks(), a.shape().first);
            std::vector<MKL_INT> ldc_array(b.num_blocks(), c.shape().first);
            std::vector<MKL_INT> group_size(b.num_blocks(), 1);

            std::vector<const double*> a_array(b.num_blocks()), b_array(b.num_blocks());
            std::vector<double*> c_array(b.num_blocks());
            std::vector<MKL_INT> n_array(b.num_blocks()), k_array(b.num_blocks()), ldb_array(b.num_blocks());

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < b.num_blocks(); ++i) {

                n_array[i] = b.get_block(i).shape().second;
                k_array[i] = b.get_block(i).shape().first;
                ldb_array[i] = b.get_block(i).shape().first;

                a_array[i] = a.data() + b.get_col_offsets()[i] * a.shape().first;
                b_array[i] = b.get_block(i).data();
                c_array[i] = c.data() + b.get_col_offsets()[i] * c.shape().first;
            }

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
                b.num_blocks(),
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const BlockDiagMatrix<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            std::vector<CBLAS_TRANSPOSE> transa_array(a.num_blocks(), transa), transb_array(a.num_blocks(), transb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            
            std::vector<float> alpha_array(a.num_blocks(), alpha);
            std::vector<float> beta_array(a.num_blocks(), beta);
            std::vector<MKL_INT> n_array(a.num_blocks(), n);
            std::vector<MKL_INT> ldb_array(a.num_blocks(), k);
            std::vector<MKL_INT> ldc_array(a.num_blocks(), m);
            std::vector<MKL_INT> group_size(a.num_blocks(), 1);

            std::vector<MKL_INT> m_array(a.num_blocks()), k_array(a.num_blocks()), lda_array(a.num_blocks());
            std::vector<const float*> a_array(a.num_blocks()), b_array(a.num_blocks());
            std::vector<float*> c_array(a.num_blocks());

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
            
                m_array[i] = a.get_block(i).shape().first;
                k_array[i] = a.get_block(i).shape().second;
                lda_array[i] = a.get_block(i).shape().first;

                a_array[i] = a.get_block(i).data();
                b_array[i] = b.data() + a.get_col_offsets()[i];
                c_array[i] = c.data() + a.get_row_offsets()[i];
            }

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
                a.num_blocks(),
                group_size.data()
            );
        }

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const BlockDiagMatrix<float>& b,
                             const float beta, Matrix_<float>& c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            std::vector<CBLAS_TRANSPOSE> transa_array(b.num_blocks(), transa), transb_array(b.num_blocks(), transb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            
            std::vector<float> alpha_array(b.num_blocks(), alpha);
            std::vector<float> beta_array(b.num_blocks(), beta);
            std::vector<MKL_INT> m_array(b.num_blocks(), a.shape().first);
            std::vector<MKL_INT> lda_array(b.num_blocks(), a.shape().first);
            std::vector<MKL_INT> ldc_array(b.num_blocks(), c.shape().first);
            std::vector<MKL_INT> group_size(b.num_blocks(), 1);

            std::vector<const float*> a_array(b.num_blocks()), b_array(b.num_blocks());
            std::vector<float*> c_array(b.num_blocks());
            std::vector<MKL_INT> n_array(b.num_blocks()), k_array(b.num_blocks()), ldb_array(b.num_blocks());

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < b.num_blocks(); ++i) {

                n_array[i] = b.get_block(i).shape().second;
                k_array[i] = b.get_block(i).shape().first;
                ldb_array[i] = b.get_block(i).shape().first;

                a_array[i] = a.data() + b.get_col_offsets()[i] * a.shape().first;
                b_array[i] = b.get_block(i).data();
                c_array[i] = c.data() + b.get_col_offsets()[i] * c.shape().first;
            }

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
                b.num_blocks(),
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