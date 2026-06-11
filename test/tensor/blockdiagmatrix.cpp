#include "../common.h"
#include <iostream>

template<typename T>
using CPUBlockDiagMatrix = lahva::cpu::BlockDiagMatrix<T>;

template<typename T>
using CPUMatrix = lahva::cpu::Matrix<T>;

template<typename T>
using CPUVector = lahva::cpu::Vector<T>;

#ifdef _CUDA
using namespace lahva::gpu;

template<typename T>
using GPUBlockDiagMatrix = BlockDiagMatrix<T>;

template<typename T>
using GPUMatrix = Matrix<T>;

template<typename T>
using GPUVector = lahva::cpu::Vector<T, CudaHostAllocator<T>>;
#endif

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_single_block()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(3, 3), {1, 2, 3, 4, 5, 6, 7, 8, 9}));

    int fail = 0;
    if (m(0, 0) != T{1}) { std::cout << "FAIL single_block (0,0)\n"; fail++; }
    if (m(0, 2) != T{7}) { std::cout << "FAIL single_block (0,2)\n"; fail++; }
    if (m(1, 0) != T{2}) { std::cout << "FAIL single_block (1,0)\n"; fail++; }
    if (m(1, 1) != T{5}) { std::cout << "FAIL single_block (1,1)\n"; fail++; }
    if (m(2, 2) != T{9}) { std::cout << "FAIL single_block (2,2)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_two_blocks()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    int fail = 0;

    // Block 0 elements
    if (m(0, 0) != T{1}) { std::cout << "FAIL two_blocks block0 (0,0)\n"; fail++; }
    if (m(0, 1) != T{3}) { std::cout << "FAIL two_blocks block0 (0,1)\n"; fail++; }
    if (m(1, 0) != T{2}) { std::cout << "FAIL two_blocks block0 (1,0)\n"; fail++; }
    if (m(1, 1) != T{4}) { std::cout << "FAIL two_blocks block0 (1,1)\n"; fail++; }

    // Block 1 elements (column-major: col0={10,20,30}, col1={40,50,60}, col2={70,80,90})
    if (m(2, 2) != T{10}) { std::cout << "FAIL two_blocks block1 (2,2)\n"; fail++; }
    if (m(2, 4) != T{70}) { std::cout << "FAIL two_blocks block1 (2,4)\n"; fail++; }
    if (m(3, 2) != T{20}) { std::cout << "FAIL two_blocks block1 (3,2)\n"; fail++; }
    if (m(3, 3) != T{50}) { std::cout << "FAIL two_blocks block1 (3,3)\n"; fail++; }
    if (m(4, 4) != T{90}) { std::cout << "FAIL two_blocks block1 (4,4)\n"; fail++; }

    // Off-block zero regions
    if (m(0, 2) != T{0}) { std::cout << "FAIL two_blocks zero (0,2)\n"; fail++; }
    if (m(0, 4) != T{0}) { std::cout << "FAIL two_blocks zero (0,4)\n"; fail++; }
    if (m(1, 3) != T{0}) { std::cout << "FAIL two_blocks zero (1,3)\n"; fail++; }
    if (m(3, 0) != T{0}) { std::cout << "FAIL two_blocks zero (3,0)\n"; fail++; }
    if (m(4, 1) != T{0}) { std::cout << "FAIL two_blocks zero (4,1)\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_three_blocks()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(1, 1), {7}));
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(1, 1), {9}));

    int fail = 0;

    if (m(0, 0) != T{7}) { std::cout << "FAIL three_blocks block0 (0,0)\n"; fail++; }
    if (m(1, 1) != T{1}) { std::cout << "FAIL three_blocks block1 (1,1)\n"; fail++; }
    if (m(1, 2) != T{3}) { std::cout << "FAIL three_blocks block1 (1,2)\n"; fail++; }
    if (m(2, 1) != T{2}) { std::cout << "FAIL three_blocks block1 (2,1)\n"; fail++; }
    if (m(2, 2) != T{4}) { std::cout << "FAIL three_blocks block1 (2,2)\n"; fail++; }
    if (m(3, 3) != T{9}) { std::cout << "FAIL three_blocks block2 (3,3)\n"; fail++; }

    // Off-block zeros
    if (m(0, 1) != T{0}) { std::cout << "FAIL three_blocks zero (0,1)\n"; fail++; }
    if (m(1, 0) != T{0}) { std::cout << "FAIL three_blocks zero (1,0)\n"; fail++; }
    if (m(0, 3) != T{0}) { std::cout << "FAIL three_blocks zero (0,3)\n"; fail++; }
    if (m(2, 3) != T{0}) { std::cout << "FAIL three_blocks zero (2,3)\n"; fail++; }
    if (m(3, 2) != T{0}) { std::cout << "FAIL three_blocks zero (3,2)\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_scalar_add()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(2, 2), {10, 20, 30, 40}));
    m += T{5};

    int fail = 0;

    if (m(0, 0) != T{6})  { std::cout << "FAIL scalar_add block0 (0,0)\n"; fail++; }
    if (m(0, 1) != T{8})  { std::cout << "FAIL scalar_add block0 (0,1)\n"; fail++; }
    if (m(1, 0) != T{7})  { std::cout << "FAIL scalar_add block0 (1,0)\n"; fail++; }
    if (m(1, 1) != T{9})  { std::cout << "FAIL scalar_add block0 (1,1)\n"; fail++; }
    if (m(2, 2) != T{15}) { std::cout << "FAIL scalar_add block1 (2,2)\n"; fail++; }
    if (m(2, 3) != T{35}) { std::cout << "FAIL scalar_add block1 (2,3)\n"; fail++; }
    if (m(3, 2) != T{25}) { std::cout << "FAIL scalar_add block1 (3,2)\n"; fail++; }
    if (m(3, 3) != T{45}) { std::cout << "FAIL scalar_add block1 (3,3)\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_copy_constructor()
{
    BDM<T> original;
    original.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    original.add_block(Mat<T>(Shape(2, 2), {10, 20, 30, 40}));

    BDM<T> copy(original);

    int fail = 0;
    if (copy(0, 0) != T{1})  { std::cout << "FAIL copy_ctor (0,0)\n";  fail++; }
    if (copy(1, 1) != T{4})  { std::cout << "FAIL copy_ctor (1,1)\n";  fail++; }
    if (copy(2, 2) != T{10}) { std::cout << "FAIL copy_ctor (2,2)\n";  fail++; }
    if (copy(3, 3) != T{40}) { std::cout << "FAIL copy_ctor (3,3)\n";  fail++; }
    if (copy.shape().first != 4 || copy.shape().second != 4) { std::cout << "FAIL copy_ctor shape\n"; fail++; }
    if (copy.num_blocks() != 2) { std::cout << "FAIL copy_ctor num_blocks\n"; fail++; }

    copy += T{1};
    if (original(0, 0) != T{1}) { std::cout << "FAIL copy_ctor independence\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_copy_assignment()
{
    BDM<T> original;
    original.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));

    BDM<T> copy;
    copy = original;

    int fail = 0;
    if (copy(0, 0) != T{1}) { std::cout << "FAIL copy_assign (0,0)\n"; fail++; }
    if (copy(1, 1) != T{4}) { std::cout << "FAIL copy_assign (1,1)\n"; fail++; }
    if (copy.num_blocks() != 1) { std::cout << "FAIL copy_assign num_blocks\n"; fail++; }

    copy += T{1};
    if (original(0, 0) != T{1}) { std::cout << "FAIL copy_assign independence\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_move_constructor()
{
    BDM<T> original;
    original.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));

    BDM<T> moved(std::move(original));

    int fail = 0;
    if (moved(0, 0) != T{1}) { std::cout << "FAIL move_ctor (0,0)\n"; fail++; }
    if (moved(1, 1) != T{4}) { std::cout << "FAIL move_ctor (1,1)\n"; fail++; }
    if (moved.shape().first != 2 || moved.shape().second != 2) { std::cout << "FAIL move_ctor shape\n"; fail++; }
    if (original.num_blocks() != 0) { std::cout << "FAIL move_ctor source not empty\n"; fail++; }
    if (original.shape().first != 0 || original.shape().second != 0) { std::cout << "FAIL move_ctor source shape\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_move_assignment()
{
    BDM<T> original;
    original.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));

    BDM<T> moved;
    moved = std::move(original);

    int fail = 0;
    if (moved(0, 0) != T{1}) { std::cout << "FAIL move_assign (0,0)\n"; fail++; }
    if (moved(1, 1) != T{4}) { std::cout << "FAIL move_assign (1,1)\n"; fail++; }
    if (original.num_blocks() != 0) { std::cout << "FAIL move_assign source not empty\n"; fail++; }
    if (original.shape().first != 0 || original.shape().second != 0) { std::cout << "FAIL move_assign source shape\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat,
         template<typename> class Vec>
int test_get_diagonal()
{
    BDM<T> m;
    // 2x2 column-major {1,2,3,4}: diag = [1, 4]
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    // 3x3 column-major {10..90}: diag = [10, 50, 90]
    m.add_block(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    Vec<T> diag = m.get_diagonal();

    int fail = 0;
    if (diag.size() != 5)  { std::cout << "FAIL get_diagonal size\n";  fail++; }
    if (diag[0] != T{1})   { std::cout << "FAIL get_diagonal [0]\n";   fail++; }
    if (diag[1] != T{4})   { std::cout << "FAIL get_diagonal [1]\n";   fail++; }
    if (diag[2] != T{10})  { std::cout << "FAIL get_diagonal [2]\n";   fail++; }
    if (diag[3] != T{50})  { std::cout << "FAIL get_diagonal [3]\n";   fail++; }
    if (diag[4] != T{90})  { std::cout << "FAIL get_diagonal [4]\n";   fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat,
         template<typename> class Vec>
int test_set_diagonal()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    Vec<T> new_diag({100, 200, 300, 400, 500});
    m.set_diagonal(new_diag);

    int fail = 0;
    if (m(0, 0) != T{100}) { std::cout << "FAIL set_diagonal (0,0)\n"; fail++; }
    if (m(1, 1) != T{200}) { std::cout << "FAIL set_diagonal (1,1)\n"; fail++; }
    if (m(2, 2) != T{300}) { std::cout << "FAIL set_diagonal (2,2)\n"; fail++; }
    if (m(3, 3) != T{400}) { std::cout << "FAIL set_diagonal (3,3)\n"; fail++; }
    if (m(4, 4) != T{500}) { std::cout << "FAIL set_diagonal (4,4)\n"; fail++; }
    // off-diagonal (but on-block) elements unchanged
    if (m(1, 0) != T{2})   { std::cout << "FAIL set_diagonal off-diag (1,0)\n"; fail++; }
    if (m(0, 1) != T{3})   { std::cout << "FAIL set_diagonal off-diag (0,1)\n"; fail++; }
    if (m(3, 2) != T{20})  { std::cout << "FAIL set_diagonal off-diag (3,2)\n"; fail++; }

    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_symmetrize()
{
    // 2x2 column-major {1,2,3,4}: [[1,3],[2,4]]
    // After symmetrize: [[1,2.5],[2.5,4]]
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    // 3x3 column-major: [[10,40,70],[20,50,80],[30,60,90]]
    // After symmetrize: [[10,30,50],[30,50,70],[50,70,90]]
    m.add_block(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));
    m.symmetrize();

    int fail = 0;
    if (m(0, 0) != T{1})    { std::cout << "FAIL symmetrize block0 (0,0)\n"; fail++; }
    if (m(1, 1) != T{4})    { std::cout << "FAIL symmetrize block0 (1,1)\n"; fail++; }
    if (m(0, 1) != T{2.5})  { std::cout << "FAIL symmetrize block0 (0,1)\n"; fail++; }
    if (m(1, 0) != T{2.5})  { std::cout << "FAIL symmetrize block0 (1,0)\n"; fail++; }
    if (m(2, 2) != T{10})   { std::cout << "FAIL symmetrize block1 (2,2)\n"; fail++; }
    if (m(3, 3) != T{50})   { std::cout << "FAIL symmetrize block1 (3,3)\n"; fail++; }
    if (m(4, 4) != T{90})   { std::cout << "FAIL symmetrize block1 (4,4)\n"; fail++; }
    if (m(2, 3) != T{30})   { std::cout << "FAIL symmetrize block1 (2,3)\n"; fail++; }
    if (m(3, 2) != T{30})   { std::cout << "FAIL symmetrize block1 (3,2)\n"; fail++; }
    if (m(2, 4) != T{50})   { std::cout << "FAIL symmetrize block1 (2,4)\n"; fail++; }
    if (m(4, 2) != T{50})   { std::cout << "FAIL symmetrize block1 (4,2)\n"; fail++; }
    if (m(3, 4) != T{70})   { std::cout << "FAIL symmetrize block1 (3,4)\n"; fail++; }
    if (m(4, 3) != T{70})   { std::cout << "FAIL symmetrize block1 (4,3)\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL symmetrize zero off-block (0,2)\n"; fail++; }
    if (m(4, 0) != T{0}) { std::cout << "FAIL symmetrize zero off-block (4,0)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat,
         template<typename> class Vec>
int test_get_diagonal_runtime()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    lahva::CPURuntime rt;
    Vec<T> diag = m.get_diagonal(rt);

    int fail = 0;
    if (diag.size() != 5) { std::cout << "FAIL get_diagonal_runtime size\n"; fail++; }
    if (diag[0] != T{1})  { std::cout << "FAIL get_diagonal_runtime [0]\n"; fail++; }
    if (diag[1] != T{4})  { std::cout << "FAIL get_diagonal_runtime [1]\n"; fail++; }
    if (diag[2] != T{10}) { std::cout << "FAIL get_diagonal_runtime [2]\n"; fail++; }
    if (diag[3] != T{50}) { std::cout << "FAIL get_diagonal_runtime [3]\n"; fail++; }
    if (diag[4] != T{90}) { std::cout << "FAIL get_diagonal_runtime [4]\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat,
         template<typename> class Vec>
int test_set_diagonal_runtime()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    m.add_block(Mat<T>(Shape(2, 2), {10, 20, 30, 40}));

    lahva::CPURuntime rt;
    const Vec<T> new_diag({100, 200, 300, 400});
    m.set_diagonal(rt, new_diag);

    int fail = 0;
    if (m(0, 0) != T{100}) { std::cout << "FAIL set_diagonal_runtime (0,0)\n"; fail++; }
    if (m(1, 1) != T{200}) { std::cout << "FAIL set_diagonal_runtime (1,1)\n"; fail++; }
    if (m(2, 2) != T{300}) { std::cout << "FAIL set_diagonal_runtime (2,2)\n"; fail++; }
    if (m(3, 3) != T{400}) { std::cout << "FAIL set_diagonal_runtime (3,3)\n"; fail++; }
    if (m(1, 0) != T{2})   { std::cout << "FAIL set_diagonal_runtime off-diag (1,0)\n"; fail++; }
    if (m(0, 1) != T{3})   { std::cout << "FAIL set_diagonal_runtime off-diag (0,1)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_symmetrize_runtime()
{
    BDM<T> m;
    m.add_block(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));

    lahva::CPURuntime rt;
    m.symmetrize(rt);

    int fail = 0;
    if (m(0, 0) != T{1})   { std::cout << "FAIL symmetrize_runtime (0,0)\n"; fail++; }
    if (m(1, 1) != T{4})   { std::cout << "FAIL symmetrize_runtime (1,1)\n"; fail++; }
    if (m(0, 1) != T{2.5}) { std::cout << "FAIL symmetrize_runtime (0,1)\n"; fail++; }
    if (m(1, 0) != T{2.5}) { std::cout << "FAIL symmetrize_runtime (1,0)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_homogeneous_ctor()
{
    BDM<T> m(3, Shape(2, 2));

    int fail = 0;
    if (m.shape().first != 6 || m.shape().second != 6) { std::cout << "FAIL homogeneous_ctor shape\n"; fail++; }
    if (m.num_blocks() != 3) { std::cout << "FAIL homogeneous_ctor num_blocks\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL homogeneous_ctor zero off-block (0,2)\n"; fail++; }
    if (m(2, 4) != T{0}) { std::cout << "FAIL homogeneous_ctor zero off-block (2,4)\n"; fail++; }
    m(0, 0) = T{1}; m(4, 4) = T{2};
    if (m(0, 0) != T{1}) { std::cout << "FAIL homogeneous_ctor write (0,0)\n"; fail++; }
    if (m(4, 4) != T{2}) { std::cout << "FAIL homogeneous_ctor write (4,4)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_homogeneous_val_ctor()
{
    BDM<T> m(4, Shape(3, 2), T{5});

    int fail = 0;
    if (m.shape().first != 12 || m.shape().second != 8) { std::cout << "FAIL homogeneous_val_ctor shape\n"; fail++; }
    if (m.num_blocks() != 4) { std::cout << "FAIL homogeneous_val_ctor num_blocks\n"; fail++; }
    if (m(0, 0)  != T{5}) { std::cout << "FAIL homogeneous_val_ctor (0,0)\n"; fail++; }
    if (m(2, 1)  != T{5}) { std::cout << "FAIL homogeneous_val_ctor (2,1)\n"; fail++; }
    if (m(11, 7) != T{5}) { std::cout << "FAIL homogeneous_val_ctor (11,7)\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL homogeneous_val_ctor zero off-block (0,2)\n"; fail++; }
    if (m(3, 0) != T{0}) { std::cout << "FAIL homogeneous_val_ctor zero off-block (3,0)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_shapes_ctor()
{
    std::vector<Shape> shapes = {Shape(2, 2), Shape(3, 3)};
    BDM<T> m(shapes);

    int fail = 0;
    if (m.shape().first != 5 || m.shape().second != 5) { std::cout << "FAIL shapes_ctor shape\n"; fail++; }
    if (m.num_blocks() != 2) { std::cout << "FAIL shapes_ctor num_blocks\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL shapes_ctor zero off-block (0,2)\n"; fail++; }
    if (m(3, 1) != T{0}) { std::cout << "FAIL shapes_ctor zero off-block (3,1)\n"; fail++; }
    m(0, 0) = T{42};
    if (m(0, 0) != T{42}) { std::cout << "FAIL shapes_ctor write (0,0)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_shapes_val_ctor()
{
    std::vector<Shape> shapes = {Shape(2, 2), Shape(3, 2)};
    BDM<T> m(shapes, T{7});

    int fail = 0;
    if (m.shape().first != 5 || m.shape().second != 4) { std::cout << "FAIL shapes_val_ctor shape\n"; fail++; }
    if (m.num_blocks() != 2) { std::cout << "FAIL shapes_val_ctor num_blocks\n"; fail++; }
    if (m(0, 0) != T{7}) { std::cout << "FAIL shapes_val_ctor (0,0)\n"; fail++; }
    if (m(1, 1) != T{7}) { std::cout << "FAIL shapes_val_ctor (1,1)\n"; fail++; }
    if (m(2, 2) != T{7}) { std::cout << "FAIL shapes_val_ctor (2,2)\n"; fail++; }
    if (m(4, 3) != T{7}) { std::cout << "FAIL shapes_val_ctor (4,3)\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL shapes_val_ctor zero off-block (0,2)\n"; fail++; }
    if (m(2, 0) != T{0}) { std::cout << "FAIL shapes_val_ctor zero off-block (2,0)\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_from_matrix_vec_copy()
{
    std::vector<Mat<T>> blocks;
    blocks.push_back(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    blocks.push_back(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    BDM<T> m(blocks);

    int fail = 0;
    if (m.shape().first != 5 || m.shape().second != 5) { std::cout << "FAIL from_vec_copy shape\n"; fail++; }
    if (m.num_blocks() != 2)  { std::cout << "FAIL from_vec_copy num_blocks\n"; fail++; }
    if (m(0, 0) != T{1})  { std::cout << "FAIL from_vec_copy (0,0)\n"; fail++; }
    if (m(1, 1) != T{4})  { std::cout << "FAIL from_vec_copy (1,1)\n"; fail++; }
    if (m(2, 2) != T{10}) { std::cout << "FAIL from_vec_copy (2,2)\n"; fail++; }
    if (m(4, 4) != T{90}) { std::cout << "FAIL from_vec_copy (4,4)\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL from_vec_copy zero off-block (0,2)\n"; fail++; }
    if (m(3, 0) != T{0}) { std::cout << "FAIL from_vec_copy zero off-block (3,0)\n"; fail++; }
    if (blocks[0](0, 0) != T{1}) { std::cout << "FAIL from_vec_copy source mutated\n"; fail++; }
    return fail;
}

template<typename T, template<typename> class BDM, template<typename> class Mat>
int test_from_matrix_vec_move()
{
    std::vector<Mat<T>> blocks;
    blocks.push_back(Mat<T>(Shape(2, 2), {1, 2, 3, 4}));
    blocks.push_back(Mat<T>(Shape(3, 3), {10, 20, 30, 40, 50, 60, 70, 80, 90}));

    BDM<T> m(std::move(blocks));

    int fail = 0;
    if (m.shape().first != 5 || m.shape().second != 5) { std::cout << "FAIL from_vec_move shape\n"; fail++; }
    if (m.num_blocks() != 2)  { std::cout << "FAIL from_vec_move num_blocks\n"; fail++; }
    if (m(0, 0) != T{1})  { std::cout << "FAIL from_vec_move (0,0)\n"; fail++; }
    if (m(1, 1) != T{4})  { std::cout << "FAIL from_vec_move (1,1)\n"; fail++; }
    if (m(2, 2) != T{10}) { std::cout << "FAIL from_vec_move (2,2)\n"; fail++; }
    if (m(4, 4) != T{90}) { std::cout << "FAIL from_vec_move (4,4)\n"; fail++; }
    if (m(0, 2) != T{0}) { std::cout << "FAIL from_vec_move zero off-block (0,2)\n"; fail++; }
    return fail;
}

// ---- helpers to call all tests for one backend -----------------------------

template<typename T, template<typename> class BDM, template<typename> class Mat,
         template<typename> class Vec>
int run_all(const char* backend, const char* scalar)
{
    int stat = 0;
    auto run = [&](const char* name, int v) {
        std::cout << name << " (" << backend << ", " << scalar << ")\n";
        stat += v;
    };
    run("single_block",           test_single_block<T, BDM, Mat>());
    run("two_blocks",             test_two_blocks<T, BDM, Mat>());
    run("three_blocks",           test_three_blocks<T, BDM, Mat>());
    run("scalar_add",             test_scalar_add<T, BDM, Mat>());
    run("copy_constructor",       test_copy_constructor<T, BDM, Mat>());
    run("copy_assignment",        test_copy_assignment<T, BDM, Mat>());
    run("move_constructor",       test_move_constructor<T, BDM, Mat>());
    run("move_assignment",        test_move_assignment<T, BDM, Mat>());
    run("get_diagonal",           test_get_diagonal<T, BDM, Mat, Vec>());
    run("set_diagonal",           test_set_diagonal<T, BDM, Mat, Vec>());
    run("symmetrize",             test_symmetrize<T, BDM, Mat>());
    run("get_diagonal_runtime",   test_get_diagonal_runtime<T, BDM, Mat, Vec>());
    run("set_diagonal_runtime",   test_set_diagonal_runtime<T, BDM, Mat, Vec>());
    run("symmetrize_runtime",     test_symmetrize_runtime<T, BDM, Mat>());
    run("homogeneous_ctor",       test_homogeneous_ctor<T, BDM, Mat>());
    run("homogeneous_val_ctor",   test_homogeneous_val_ctor<T, BDM, Mat>());
    run("shapes_ctor",            test_shapes_ctor<T, BDM, Mat>());
    run("shapes_val_ctor",        test_shapes_val_ctor<T, BDM, Mat>());
    run("from_matrix_vec_copy",   test_from_matrix_vec_copy<T, BDM, Mat>());
    run("from_matrix_vec_move",   test_from_matrix_vec_move<T, BDM, Mat>());
    return stat;
}

int main()
{
    int stat = 0;

    stat += run_all<double, CPUBlockDiagMatrix, CPUMatrix, CPUVector>("cpu", "double");
    stat += run_all<float,  CPUBlockDiagMatrix, CPUMatrix, CPUVector>("cpu", "float");

#ifdef _CUDA
    stat += run_all<double, GPUBlockDiagMatrix, GPUMatrix, GPUVector>("gpu", "double");
    stat += run_all<float,  GPUBlockDiagMatrix, GPUMatrix, GPUVector>("gpu", "float");
#endif

    return stat;
}
