#pragma once
#include "stdlib.h"

// template <size_t n_vecs, size_t dim>
// size_t get_index(const size_t &index)
// {
//     return get_index(n_vecs, dim, index);
// }
size_t get_index(const size_t &n_vecs, const size_t &dim, const size_t &index)
{
    return n_vecs * dim + index;
}