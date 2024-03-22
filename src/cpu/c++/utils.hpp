#include "linalg.hpp"

#include <assert.h>

namespace tcgmtensor{
    template<typename T>
    void check_equal_size(const vector<T> v1, const vector<T> v2){
        assert(v1.size() == v2.size());
    }
    }