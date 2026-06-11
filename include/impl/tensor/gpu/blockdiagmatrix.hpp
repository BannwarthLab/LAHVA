#pragma once
#include "runtime.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/sparse_format.hpp"
#include "impl/tensor/cpu/blockdiagmatrix.hpp"
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace lahva
{
    namespace gpu
    {
        // Forward declaration
        template <typename T>
        class BlockDiagMatrixSparse;
        // Helper struct for GPU block diagonal matrix data
        template<typename T>
        struct GPUBlockDiagData {
            T *d_data;           // Device pointer to padded packed blocks
            size_t padded_stride; // Stride for padded blocks
            size_t max_m;        // Maximum block row dimension
            size_t max_k;        // Maximum block column dimension
            int num_blocks;      // Number of blocks
            T *h_packed;         // Host pinned memory (for cleanup)
        };
    
    
    template<typename T>
    class BlockDiagMatrix_ : public virtual GPUTensor_<T>, virtual public gpu::LowTriMatrix_<T>
    {
        public:
            virtual Shape shape() const  = 0;
    };

    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class BlockDiagMatrix : virtual public GPUTensor<T, Allocator, GPUAllocator>, virtual public BlockDiagMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
        using gpualloc_ptr = GPUAllocator_<T>;
    protected:
        // Total matrix shape in each dimension
        size_t n_rows_ = 0;
        size_t n_cols_ = 0;
        std::vector<Matrix<T, Allocator>> matrices;
        std::vector<size_t> block_rows_;
        std::vector<size_t> block_cols_;
        std::vector<int> row_offsets_;  // Cumulative row offsets for each block
        std::vector<int> col_offsets_;  // Cumulative column offsets for each block
        
        // GPU data cache (mutable for lazy initialization in const methods)
        mutable GPUBlockDiagData<T> *gpu_data_ = nullptr;
        mutable bool gpu_data_valid_ = false;

    public:
        BlockDiagMatrix() {};
        virtual ~BlockDiagMatrix() {
            if (gpu_data_ != nullptr) {
                free_gpu_data(*gpu_data_);
                delete gpu_data_;
            }
        }

        BlockDiagMatrix(const BlockDiagMatrix &other) :
            n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
            matrices{other.matrices},
            block_rows_{other.block_rows_},
            block_cols_{other.block_cols_},
            row_offsets_{other.row_offsets_},
            col_offsets_{other.col_offsets_}
        {}

        BlockDiagMatrix &operator=(const BlockDiagMatrix &other)
        {
            if (this != &other)
            {
                free_gpu_cache();
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                matrices = other.matrices;
                block_rows_ = other.block_rows_;
                block_cols_ = other.block_cols_;
                row_offsets_ = other.row_offsets_;
                col_offsets_ = other.col_offsets_;
            }
            return *this;
        }

        BlockDiagMatrix(BlockDiagMatrix &&other) noexcept :
            n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
            matrices{std::move(other.matrices)},
            block_rows_{std::move(other.block_rows_)},
            block_cols_{std::move(other.block_cols_)},
            row_offsets_{std::move(other.row_offsets_)},
            col_offsets_{std::move(other.col_offsets_)},
            gpu_data_{other.gpu_data_},
            gpu_data_valid_{other.gpu_data_valid_}
        {
            other.n_rows_ = 0;
            other.n_cols_ = 0;
            other.gpu_data_ = nullptr;
            other.gpu_data_valid_ = false;
        }

        BlockDiagMatrix &operator=(BlockDiagMatrix &&other) noexcept
        {
            if (this != &other)
            {
                free_gpu_cache();
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                matrices = std::move(other.matrices);
                block_rows_ = std::move(other.block_rows_);
                block_cols_ = std::move(other.block_cols_);
                row_offsets_ = std::move(other.row_offsets_);
                col_offsets_ = std::move(other.col_offsets_);
                gpu_data_ = other.gpu_data_;
                gpu_data_valid_ = other.gpu_data_valid_;
                other.n_rows_ = 0;
                other.n_cols_ = 0;
                other.gpu_data_ = nullptr;
                other.gpu_data_valid_ = false;
            }
            return *this;
        }

        BlockDiagMatrix(size_t n_blocks, Shape block_shape)
        {
            matrices.reserve(n_blocks);
            block_rows_.reserve(n_blocks);
            block_cols_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape);
            build_offsets_();
        }

        BlockDiagMatrix(size_t n_blocks, Shape block_shape, T val)
        {
            matrices.reserve(n_blocks);
            block_rows_.reserve(n_blocks);
            block_cols_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape, val);
            build_offsets_();
        }

        explicit BlockDiagMatrix(const std::vector<Shape> &shapes)
        {
            matrices.reserve(shapes.size());
            block_rows_.reserve(shapes.size());
            block_cols_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s);
            build_offsets_();
        }

        BlockDiagMatrix(const std::vector<Shape> &shapes, T val)
        {
            matrices.reserve(shapes.size());
            block_rows_.reserve(shapes.size());
            block_cols_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s, val);
            build_offsets_();
        }

        explicit BlockDiagMatrix(const std::vector<Matrix<T, Allocator>> &blocks)
            : matrices(blocks)
        {
            block_rows_.reserve(matrices.size());
            block_cols_.reserve(matrices.size());
            build_offsets_();
        }

        explicit BlockDiagMatrix(std::vector<Matrix<T, Allocator>> &&blocks)
            : matrices(std::move(blocks))
        {
            block_rows_.reserve(matrices.size());
            block_cols_.reserve(matrices.size());
            build_offsets_();
        }

        //! @brief in-place, scalar addition
        BlockDiagMatrix &operator+=(T val)
        {
            free_gpu_cache();
            for (auto &m : matrices)
                m += val;
            return *this;
        }

        //! @return number of rows/columns of the Matrix
        Shape shape() const { return Shape{n_rows_, n_cols_}; }

        //! @return number of rows/columns of the submatrices
        std::vector<std::vector<size_t>> block_shapes() const
            { 
                std::vector<std::vector<size_t>> block_shapes;
                for (size_t i = 0; i < matrices.size(); i++) {
                    Shape s = matrices[i].shape();
                    block_shapes.push_back({s.first, s.second});
                }
                return block_shapes;
            }
        
        //! @return number of blocks
        size_t num_blocks() const { return matrices.size(); }
        
        //! @return const reference to a block matrix
        const Matrix<T, Allocator>& get_block(size_t idx) const { return matrices[idx]; }
        
        //! @return const reference to row offsets (cumulative row position of each block)
        const std::vector<int>& get_row_offsets() const { return row_offsets_; }
        
        //! @return const reference to column offsets (cumulative column position of each block)
        const std::vector<int>& get_col_offsets() const { return col_offsets_; }

        //! @return const reference to block row sizes
        const std::vector<size_t>& get_block_rows() const { return block_rows_; }

        //! @return const reference to block column sizes
        const std::vector<size_t>& get_block_cols() const { return block_cols_; }

        //! @return concatenated diagonals of all blocks
        cpu::Vector<T, Allocator> get_diagonal() const
        {
            size_t total = 0;
            for (const auto &m : matrices)
            {
                Shape s = m.shape();
                total += std::min(s.first, s.second);
            }
            cpu::Vector<T, Allocator> diag(total);
            size_t offset = 0;
            for (const auto &m : matrices)
            {
                Shape s = m.shape();
                size_t min_dim = std::min(s.first, s.second);
                for (size_t i = 0; i < min_dim; i++)
                    diag[offset + i] = m(i, i);
                offset += min_dim;
            }
            return diag;
        }

        //! @brief set diagonal of each block from a flat vector of concatenated block diagonals
        void set_diagonal(const cpu::Vector<T, Allocator> &diag)
        {
            free_gpu_cache();
            size_t offset = 0;
            for (auto &m : matrices)
            {
                Shape s = m.shape();
                size_t min_dim = std::min(s.first, s.second);
#pragma omp for
                for (size_t i = 0; i < min_dim; i++)
                    m(i, i) = diag[offset + i];
                offset += min_dim;
            }
        }

        //! @brief symmetrize each block in-place: block = (block + block^T) / 2
        void symmetrize()
        {
            free_gpu_cache();
            for (auto &m : matrices)
                m.symmetrize();
        }

        template <typename... Args>
        void symmetrize(const CPURuntime &rt_, Args &&...args)
        {
            (symmetrize(args...));
        }

        template <typename... Args>
        cpu::Vector<T, Allocator> get_diagonal(const CPURuntime &rt_, Args &&...args)
        {
            return get_diagonal(args...);
        }

        template <typename... Args>
        void set_diagonal(const CPURuntime &rt_, Args &&...args)
        {
            (set_diagonal(args...));
        }

        //! @return const void pointer to block data (for compatibility with abstract interface)
        const void* get_block_data(size_t idx) const {
            return static_cast<const void*>(matrices[idx].data());
        }

        virtual T &operator()(size_t i, size_t j) override {
            for (size_t b = 0; b < matrices.size(); ++b) {
                if (i >= row_offsets_[b] && i < row_offsets_[b] + (int)matrices[b].shape().first &&
                    j >= col_offsets_[b] && j < col_offsets_[b] + (int)matrices[b].shape().second) {
                    return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
                }
            }
            static T zero{};
            return zero;
        }

        virtual const T &operator()(size_t i, size_t j) const override {
            for (size_t b = 0; b < matrices.size(); ++b) {
                if (i >= row_offsets_[b] && i < row_offsets_[b] + (int)matrices[b].shape().first &&
                    j >= col_offsets_[b] && j < col_offsets_[b] + (int)matrices[b].shape().second) {
                    return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
                }
            }
            static const T zero{};
            return zero;
        }

        void add_block(const Matrix<T, Allocator> &block) {
            matrices.push_back(block);
            block_rows_.push_back(block.shape().first);
            block_cols_.push_back(block.shape().second);
            n_rows_ += block.shape().first;
            n_cols_ += block.shape().second;
            
            // Update offsets
            if (col_offsets_.empty()) {
                col_offsets_.push_back(0);
                row_offsets_.push_back(0);
            }
            col_offsets_.push_back(col_offsets_.back() + block.shape().second);
            row_offsets_.push_back(row_offsets_.back() + block.shape().first);
        }




        //! prints the Matrix as string
        void print() const {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print();
            }
        }

        void print(const char* file) const
        {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print(file);
            }
        }

        //! @brief Ensure block diagonal matrix is on GPU with optimal packing
        //! Uses caching - first call copies to GPU, subsequent calls return cached data
        const GPUBlockDiagData<T>& ensure_on_gpu() const {
            if (!gpu_data_valid_) {
                copy_to_gpu_impl();
            }
            return *gpu_data_;
        }
        
        //! @brief Check if matrix data is currently cached on GPU
        bool is_on_gpu() const {
            return gpu_data_valid_;
        }
        
        //! @brief Free GPU cache and allow re-copy on next ensure_on_gpu()
        void free_gpu_cache() {
            if (gpu_data_ != nullptr) {
                free_gpu_data(*gpu_data_);
                delete gpu_data_;
                gpu_data_ = nullptr;
            }
            gpu_data_valid_ = false;
        }
        
    private:
        static void check_size_(size_t n_rows, size_t n_cols)
        {
            if (n_rows != 0 && n_cols > SIZE_MAX / n_rows)
                throw std::out_of_range("Block size exceeds maximum representable size.");
        }

        void build_offsets_()
        {
            row_offsets_.reserve(matrices.size() + 1);
            col_offsets_.reserve(matrices.size() + 1);
            row_offsets_.push_back(0);
            col_offsets_.push_back(0);
            for (const auto &m : matrices)
            {
                Shape s = m.shape();
                n_rows_ += s.first;
                n_cols_ += s.second;
                block_rows_.push_back(s.first);
                block_cols_.push_back(s.second);
                row_offsets_.push_back(row_offsets_.back() + static_cast<int>(s.first));
                col_offsets_.push_back(col_offsets_.back() + static_cast<int>(s.second));
            }
        }

        //! @brief Internal method to copy block diagonal matrix to GPU
        void copy_to_gpu_impl() const {
            int num_blocks = matrices.size();
            
            // Find max dimensions for padding
            size_t max_m = *std::max_element(block_rows_.begin(), block_rows_.end());
            size_t max_k = *std::max_element(block_cols_.begin(), block_cols_.end());
            
            long long padded_stride = (long long)max_m * max_k;
            
            // Allocate pinned host memory
            T *h_packed;
            get_cuda_error(cudaHostAlloc(&h_packed, padded_stride * num_blocks * sizeof(T), cudaHostAllocDefault));
            
            // Initialize to zero
            std::memset(h_packed, 0, padded_stride * num_blocks * sizeof(T));
            
            // Pack blocks into contiguous layout with padding
            for (int i = 0; i < num_blocks; ++i) {
                const Matrix<T, Allocator>& block = matrices[i];
                size_t block_m = block.shape().first;
                size_t block_k = block.shape().second;
                
                // Copy block column by column into padded layout
                for (size_t j = 0; j < block_k; ++j) {
                    std::memcpy(
                        h_packed + i * padded_stride + j * max_m,
                        block.data() + j * block_m,
                        block_m * sizeof(T)
                    );
                }
            }
            
            // Allocate GPU memory and copy
            T *d_data;
            get_cuda_error(cudaMalloc(&d_data, padded_stride * num_blocks * sizeof(T)));
            get_cuda_error(cudaMemcpy(d_data, h_packed, padded_stride * num_blocks * sizeof(T), cudaMemcpyHostToDevice));
            
            // Store in cache
            gpu_data_ = new GPUBlockDiagData<T>{d_data, (size_t)padded_stride, max_m, max_k, num_blocks, h_packed};
            gpu_data_valid_ = true;
        }
        
    public:
        //! @brief Free GPU block diagonal data (static helper for manual cleanup)
        static void free_gpu_data(GPUBlockDiagData<T>& gpu_data) {
            get_cuda_error(cudaFree(gpu_data.d_data));
            get_cuda_error(cudaFreeHost(gpu_data.h_packed));
        }
        
        //! @brief Convert to sparse format on GPU
        //! @param[in] format Sparse format to use (CSR or ELLPACK, default: CSR)
        //! @return BlockDiagMatrixSparse object containing the sparse representation
        BlockDiagMatrixSparse<T> to_sparse(SparseFormat format = SparseFormat::CSR) const {
            return BlockDiagMatrixSparse<T>(*this, format);
        }
    };

    } // namespace gpu
} // namespace lahva
