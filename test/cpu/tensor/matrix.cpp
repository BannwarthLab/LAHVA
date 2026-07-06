#include "test_common.h"

using namespace lahva::cpu;

const double TOLERANCE = 5.0e-7;

// ============================================================================
// Matrix Constructor Tests
// ============================================================================

template <typename T>
int test_matrix_size_constructor() {
    int failures = 0;

    Shape shape = Shape{5, 10};
    Matrix<T> m(shape);

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Size constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 10, "Size constructor should set correct cols")) {
        failures += 1;
    }

    if (m.data() == nullptr) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_shape_constructor() {
    int failures = 0;

    Shape shape = Shape{3, 7};
    Matrix<T> m(shape);

    Shape s = m.shape();
    if (!check((int)s.first, 3, "Shape constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 7, "Shape constructor should set correct cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_size_value_constructor() {
    int failures = 0;

    Shape shape = Shape{4, 6};
    Matrix<T> m(shape, (T)2.5);

    Shape s = m.shape();
    if (!check((int)s.first, 4, "Size+value constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 6, "Size+value constructor should set correct cols")) {
        failures += 1;
    }

    // Verify all elements are initialized
    for (int i = 0; i < 4 * 6; i++) {
        if (!check((double)m.data()[i], 2.5, TOLERANCE, "All elements should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_matrix_copy_constructor() {
    int failures = 0;

    Shape shape = Shape{3, 3};
    Matrix<T> m1(shape, (T)5.0);
    Matrix<T> m2 = m1;  // Copy constructor

    Shape s1 = m1.shape();
    Shape s2 = m2.shape();

    if (!check((int)s2.first, (int)s1.first, "Copy constructor should copy rows")) {
        failures += 1;
    }

    if (!check((int)s2.second, (int)s1.second, "Copy constructor should copy cols")) {
        failures += 1;
    }

    // Verify data is copied
    for (int i = 0; i < 9; i++) {
        if (!check((double)m2.data()[i], 5.0, TOLERANCE, "Copy constructor should copy data")) {
            failures += 1;
            break;
        }
    }

    // Modify original - copy should not change
    m1.data()[0] = (T)99.0;
    if (!check((double)m2.data()[0], 5.0, TOLERANCE, "Copy should be independent")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_move_constructor() {
    int failures = 0;

    Shape shape = Shape{2, 3};
    Matrix<T> m1(shape, (T)4.0);

    Matrix<T> m2 = std::move(m1);  // Move constructor

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 2, "Move constructor should transfer rows")) {
        failures += 1;
    }

    if (!check((int)s2.second, 3, "Move constructor should transfer cols")) {
        failures += 1;
    }

    // Original should be empty
    Shape s1 = m1.shape();
    if (!check((int)s1.first, 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Matrix Attribute Tests
// ============================================================================

template <typename T>
int test_matrix_shape_attribute() {
    int failures = 0;

    Shape shape = Shape{7, 5};
    Matrix<T> m(shape);

    Shape s = m.shape();

    if (!check((int)s.first, 7, "shape().first should match rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 5, "shape().second should match cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_size_attribute() {
    int failures = 0;

    Shape shape = Shape{5, 7};
    Matrix<T> m(shape);

    // Total size should be rows * cols
    Shape s = m.shape();
    int total_size = s.first * s.second;

    if (!check(total_size, 35, "size should return rows*cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_data_access() {
    int failures = 0;

    Shape shape = Shape{3, 3};
    Matrix<T> m(shape);

    if (m.data() == nullptr) {
        failures += 1;
    }

    // Write and read
    m.data()[0] = (T)5.5;
    if (!check((double)m.data()[0], 5.5, TOLERANCE, "Data access should work")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_operator_access() {
    int failures = 0;

    Shape shape = Shape{3, 3};
    Matrix<T> m(shape);

    m(0, 0) = (T)1.0;
    m(1, 1) = (T)2.0;
    m(2, 2) = (T)3.0;

    if (!check((double)m(0, 0), 1.0, TOLERANCE, "Operator() access should work")) {
        failures += 1;
    }

    if (!check((double)m(1, 1), 2.0, TOLERANCE, "Operator() [1,1]")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Matrix Assignment Tests
// ============================================================================

template <typename T>
int test_matrix_copy_assignment() {
    int failures = 0;

    Shape shape1 = Shape{2, 3};
    Matrix<T> m1(shape1, (T)3.0);
    Matrix<T> m2;

    m2 = m1;  // Copy assignment

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 2, "Copy assignment should copy rows")) {
        failures += 1;
    }

    if (!check((int)s2.second, 3, "Copy assignment should copy cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_move_assignment() {
    int failures = 0;

    Shape shape = Shape{3, 4};
    Matrix<T> m1(shape, (T)2.0);
    Matrix<T> m2;

    m2 = std::move(m1);  // Move assignment

    Shape s2 = m2.shape();
    if (!check((int)s2.first, 3, "Move assignment should transfer rows")) {
        failures += 1;
    }

    if (!check((int)s2.second, 4, "Move assignment should transfer cols")) {
        failures += 1;
    }

    // Original should be empty
    Shape s1 = m1.shape();
    if (!check((int)s1.first, 0, "Original should be empty after move assignment")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Matrix Memory Tests
// ============================================================================

template <typename T>
int test_matrix_memory_allocation() {
    int failures = 0;

    Shape shape = Shape{100, 100};
    Matrix<T> m(shape);

    if (m.data() == nullptr) {
        failures += 1;
    }

    // Write to first and last elements
    m.data()[0] = (T)1.5;
    m.data()[9999] = (T)9999.5;

    if (!check((double)m.data()[0], 1.5, TOLERANCE, "First element should be accessible")) {
        failures += 1;
    }

    if (!check((double)m.data()[9999], 9999.5, TOLERANCE, "Last element should be accessible")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_destructor() {
    int failures = 0;

    {
        Shape shape = Shape{50, 50};
        Matrix<T> m(shape);
        if (m.data() == nullptr) {
            failures += 1;
        }
    }
    // Destructor should deallocate

    return failures;
}

// ============================================================================
// Matrix Type Tests
// ============================================================================

int test_matrix_float_type() {
    int failures = 0;

    Shape shape = Shape{3, 3};
    Matrix<float> m(shape, 2.5f);

    Shape s = m.shape();
    if (!check((int)s.first, 3, "Float matrix should have correct rows")) {
        failures += 1;
    }

    if (!check(m.data()[0], 2.5f, 1e-6f, "Float matrix should store float values")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_int_type() {
    int failures = 0;

    Shape shape = Shape{2, 2};
    Matrix<int> m(shape);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    Shape s = m.shape();
    if (!check((int)s.first, 2, "Int matrix should have correct rows")) {
        failures += 1;
    }

    if (!check(m.data()[0], 10, "Int matrix should store int values")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Matrix Operation Tests (from mat-attributes)
// ============================================================================

template <typename T>
int test_matrix_symmetrize() {
    int failures = 0;

    Shape s = Shape{3, 3};
    Matrix<T> m(s);

    m.data()[0] = (T)1.0;  m.data()[3] = (T)2.0;  m.data()[6] = (T)3.0;
    m.data()[1] = (T)4.0;  m.data()[4] = (T)5.0;  m.data()[7] = (T)6.0;
    m.data()[2] = (T)7.0;  m.data()[5] = (T)8.0;  m.data()[8] = (T)9.0;

    m.symmetrize();

    if (!check((double)m.data()[1], (double)m.data()[3], TOLERANCE, "Matrix should be symmetric after symmetrize")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_matrix_get_diagonal() {
    int failures = 0;

    Shape s = Shape{4, 4};
    Matrix<T> m(s);

    m.data()[0] = (T)1.0;   // [0,0]
    m.data()[5] = (T)2.0;   // [1,1]
    m.data()[10] = (T)3.0;  // [2,2]
    m.data()[15] = (T)4.0;  // [3,3]

    Vector<T> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, "Diagonal should have size 4")) {
        failures += 1;
    }

    T expected[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    for (int i = 0; i < 4; i++) {
        if (!check((double)diag.data()[i], (double)expected[i], TOLERANCE, "Diagonal elements should match")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_matrix_is_square() {
    int failures = 0;

    Shape s_square = Shape{5, 5};
    Matrix<T> square(s_square);

    Shape s_rect = Shape{3, 7};
    Matrix<T> rect(s_rect);

    Shape sq_shape = square.shape();
    if (sq_shape.first != sq_shape.second) {
        failures += 1;
    }

    Shape r_shape = rect.shape();
    if (r_shape.first == r_shape.second) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
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

    if (total_failures == 0) {
        std::cout << "All Matrix type tests passed!" << std::endl;
    } else {
        std::cout << "Matrix type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}
