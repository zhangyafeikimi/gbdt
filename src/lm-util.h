#ifndef GBDT_LAMBDA_MART_UTIL_H
#define GBDT_LAMBDA_MART_UTIL_H

#include <algorithm>
#include <vector>

template <class T, class Predicator>
struct SortIndicesHelper
{
    const T * const unsorted;
    const size_t unsorted_size;
    std::vector<size_t> * const indices;
    const Predicator& predicator;

    SortIndicesHelper(
        const T * _unsorted,
        size_t _unsorted_size,
        std::vector<size_t> * _indices,
        const Predicator& _predicator)
        : unsorted(_unsorted), unsorted_size(_unsorted_size), indices(_indices), predicator(_predicator) {}

    bool operator()(const size_t a, const size_t b) const
    {
        return predicator(unsorted[a], unsorted[b]);
    }
};

// sort 'unsorted' by 'predicator'
// After this function, 'unsorted' remains unchanged,
// 'unsorted[(*indices)[i]]' is sorted.
template <class T, class Predicator>
void sort_indices(
    const T * unsorted,
    size_t unsorted_size,
    std::vector<size_t> * indices,
    const Predicator& predicator)
{
    indices->clear();
    indices->reserve(unsorted_size);
    for (size_t i=0; i<unsorted_size; i++)
        indices->push_back(i);

    std::sort(indices->begin(), indices->end(),
        SortIndicesHelper<T, Predicator>(unsorted, unsorted_size, indices, predicator));
}

#endif// GBDT_LAMBDA_MART_UTIL_H
