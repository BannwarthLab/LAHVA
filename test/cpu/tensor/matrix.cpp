#include "test_common.h"

using namespace lahva::cpu;
using lahva::Shape;

// ============================================================================
// Matrix Constructor Tests
// ============================================================================

template <typename T>
int test_matrix_size_constructor() {
    Shape shape = Shape{5, 10};
    Matrix<T> m(shape);

    Shape s = m.shape();
    if (!check((int)s.first, 5, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 10, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    if (m.data() == nullptr) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_shape_constructor() {
    Shape shape = Shape{3, 7};
    Matrix<T> m(shape);

    Shape s = m.shape();
    if (!check((int)s.first, 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 7, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_size_value_constructor() {
    Shape shape = Shape{4, 6};
    Matrix<T> m(shape, (T)2.5);

    Shape s = m.shape();
    if (!check((int)s.first, 4, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 6, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Verify all elements are initialized
    for (int i = 0; i < 4 * 6; i++) {
        if (!check((double)m.data()[i], 2.5, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_matrix_copy_constructor() {
    Shape shape = Shape{3, 3};
    Matrix<T> m1(shape, (T)5.0);
    Matrix<T> m2 = m1;  // Copy constructor

    Shape s1 = m1.shape();
    Shape s2 = m2.shape();

    if (!check((int)s2.first, (int)s1.first, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s2.second, (int)s1.second, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Verify data is copied
    for (int i = 0; i < 9; i++) {
        if (!check((double)m2.data()[i], 5.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;
    }

    // Modify original - copy should not change
    m1.data()[0] = (T)99.0;
    if (!check((double)m2.data()[0], 5.0, make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_move_constructor() {
    Shape shape = Shape{2, 3};
    Matrix<T> m1(shape, (T)4.0);

    Matrix<T> m2 = std::move(m1);  // Move constructor

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 2, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s2.second, 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Original should be empty
    Shape s1 = m1.shape();
    if (!check((int)s1.first, 0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Matrix Attribute Tests
// ============================================================================

template <typename T>
int test_matrix_shape_attribute() {
    Shape shape = Shape{7, 5};
    Matrix<T> m(shape);

    Shape s = m.shape();

    if (!check((int)s.first, 7, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 5, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_size_attribute() {
    Shape shape = Shape{5, 7};
    Matrix<T> m(shape);

    // Total size should be rows * cols
    Shape s = m.shape();
    int total_size = s.first * s.second;

    if (!check(total_size, 35, make_check_msg(__func__, get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_data_access() {
    Shape shape = Shape{3, 3};
    Matrix<T> m(shape);

    if (m.data() == nullptr) return TEST_FAIL;

    // Write and read
    m.data()[0] = (T)5.5;
    if (!check((double)m.data()[0], 5.5, make_check_msg(__func__, get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_operator_access() {
    Shape shape = Shape{3, 3};
    Matrix<T> m(shape);

    m(0, 0) = (T)1.0;
    m(1, 1) = (T)2.0;
    m(2, 2) = (T)3.0;

    if (!check((double)m(0, 0), 1.0, "Operator() access should work")) return TEST_FAIL;

    if (!check((double)m(1, 1), 2.0, "Operator() [1,1]")) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Matrix Assignment Tests
// ============================================================================

template <typename T>
int test_matrix_copy_assignment() {
    Shape shape1 = Shape{2, 3};
    Matrix<T> m1(shape1, (T)3.0);
    Matrix<T> m2;

    m2 = m1;  // Copy assignment

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 2, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s2.second, 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_move_assignment() {
    Shape shape = Shape{3, 4};
    Matrix<T> m1(shape, (T)2.0);
    Matrix<T> m2;

    m2 = std::move(m1);  // Move assignment

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s2.second, 4, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Original should be empty
    Shape s1 = m1.shape();
    if (!check((int)s1.first, 0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Matrix Memory Tests
// ============================================================================

template <typename T>
int test_matrix_memory_allocation() {
    Shape shape = Shape{100, 100};
    Matrix<T> m(shape);

    if (m.data() == nullptr) return TEST_FAIL;

    // Write to first and last elements
    m.data()[0] = (T)1.5;
    m.data()[9999] = (T)9999.5;

    if (!check((double)m.data()[0], 1.5, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((double)m.data()[9999], 9999.5, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_destructor() {
    {
        Shape shape = Shape{50, 50};
        Matrix<T> m(shape);
        if (m.data() == nullptr) return TEST_FAIL;
    }
    // Destructor should deallocate

    return TEST_PASS;
}

// ============================================================================
// Matrix Type Tests
// ============================================================================

int test_matrix_float_type() {
    Shape shape = Shape{3, 3};
    Matrix<float> m(shape, 2.5f);

    Shape s = m.shape();
    if (!check((int)s.first, 3, make_check_msg(__func__, get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 2.5f, make_check_msg(__func__, get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_int_type() {
    Shape shape = Shape{2, 2};
    Matrix<int> m(shape);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    Shape s = m.shape();
    if (!check((int)s.first, 2, make_check_msg(__func__, get_type_name<int>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 10, make_check_msg(__func__, get_type_name<int>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Matrix Operation Tests (from mat-attributes)
// ============================================================================

template <typename T>
int test_matrix_symmetrize() {
    Shape s = Shape{3, 3};
    Matrix<T> m(s);

    m.data()[0] = (T)1.0;  m.data()[3] = (T)2.0;  m.data()[6] = (T)3.0;
    m.data()[1] = (T)4.0;  m.data()[4] = (T)5.0;  m.data()[7] = (T)6.0;
    m.data()[2] = (T)7.0;  m.data()[5] = (T)8.0;  m.data()[8] = (T)9.0;

    m.symmetrize();

    if (!check((double)m.data()[1], (double)m.data()[3], make_check_msg(__func__, get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_matrix_get_diagonal() {
    Shape s = Shape{4, 4};
    Matrix<T> m(s);

    m.data()[0] = (T)1.0;   // [0,0]
    m.data()[5] = (T)2.0;   // [1,1]
    m.data()[10] = (T)3.0;  // [2,2]
    m.data()[15] = (T)4.0;  // [3,3]

    Vector<T> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    T expected[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    for (int i = 0; i < 4; i++) {
        if (!check((double)diag.data()[i], (double)expected[i], make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_matrix_is_square() {
    Shape s_square = Shape{5, 5};
    Matrix<T> square(s_square);

    Shape s_rect = Shape{3, 7};
    Matrix<T> rect(s_rect);

    Shape sq_shape = square.shape();
    if (sq_shape.first != sq_shape.second) return TEST_FAIL;

    Shape r_shape = rect.shape();
    if (r_shape.first == r_shape.second) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructors - double
    total_failures += test_matrix_size_constructor<double>();
    total_failures += test_matrix_shape_constructor<double>();
    total_failures += test_matrix_size_value_constructor<double>();
    total_failures += test_matrix_copy_constructor<double>();
    total_failures += test_matrix_move_constructor<double>();

    // Constructors - float
    total_failures += test_matrix_size_constructor<float>();
    total_failures += test_matrix_shape_constructor<float>();
    total_failures += test_matrix_size_value_constructor<float>();
    total_failures += test_matrix_copy_constructor<float>();
    total_failures += test_matrix_move_constructor<float>();

    // Attributes - double
    total_failures += test_matrix_shape_attribute<double>();
    total_failures += test_matrix_size_attribute<double>();
    total_failures += test_matrix_data_access<double>();
    total_failures += test_matrix_operator_access<double>();

    // Attributes - float
    total_failures += test_matrix_shape_attribute<float>();
    total_failures += test_matrix_size_attribute<float>();
    total_failures += test_matrix_data_access<float>();
    total_failures += test_matrix_operator_access<float>();

    // Assignment - double
    total_failures += test_matrix_copy_assignment<double>();
    total_failures += test_matrix_move_assignment<double>();

    // Assignment - float
    total_failures += test_matrix_copy_assignment<float>();
    total_failures += test_matrix_move_assignment<float>();

    // Memory - double
    total_failures += test_matrix_memory_allocation<double>();
    total_failures += test_matrix_destructor<double>();

    // Memory - float
    total_failures += test_matrix_memory_allocation<float>();
    total_failures += test_matrix_destructor<float>();

    // Operations - double
    total_failures += test_matrix_symmetrize<double>();
    total_failures += test_matrix_get_diagonal<double>();
    total_failures += test_matrix_is_square<double>();

    // Operations - float
    total_failures += test_matrix_symmetrize<float>();
    total_failures += test_matrix_get_diagonal<float>();
    total_failures += test_matrix_is_square<float>();

    // Types (specific type tests - not templated)
    total_failures += test_matrix_float_type();
    total_failures += test_matrix_int_type();

    if (total_failures > 0) {
        std::cerr << "cpu/tensor/matrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/tensor/matrix tests passed!" << std::endl;
    return TEST_PASS;
}
