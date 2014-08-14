#ifndef GBDT_LAMBDA_MART_UTIL_H
#define GBDT_LAMBDA_MART_UTIL_H

#include <algorithm>
#include <vector>

template <class T, class Predicator>
struct SortIndicesHelper
{
    const std::vector<T>& unsorted;
    std::vector<size_t> * indices;
    const Predicator& predicator;

    SortIndicesHelper(
        const std::vector<T>& _unsorted,
        std::vector<size_t> * _indices,
        const Predicator& _predicator)
        : unsorted(_unsorted), indices(_indices), predicator(_predicator) {}

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
    const std::vector<T>& unsorted,
    std::vector<size_t> * indices,
    const Predicator& predicator)
{
    indices->clear();
    indices->reserve(unsorted.size());
    for (size_t i=0, s=unsorted.size(); i!=s; i++)
        indices->push_back(i);

    std::sort(indices->begin(), indices->end(),
        SortIndicesHelper<T, Predicator>(unsorted, indices, predicator));
}

// T is always double
template <class T>
class SymmetricMatrix
{
private:
    std::vector<T> m_;
    const size_t row_;

public:
    SymmetricMatrix(size_t row)
        : row_(row)
    {
        m_.resize((row_ + 1) * row_ / 2);
    }

    T& at(size_t i, size_t j)
    {
        if (j > i)
            return at(j, i);
        return m_[(i + 1) * i / 2 + j];
    }

    T at(size_t i, size_t j) const
    {
        if (j > i)
            return at(j, i);
        return m_[(i + 1) * i / 2 + j];
    }
};

#endif// GBDT_LAMBDA_MART_UTIL_H
