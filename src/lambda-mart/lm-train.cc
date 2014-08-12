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

#include <functional>

int main()
{
    std::vector<double> unsorted;
    std::vector<size_t> indices;

    unsorted.push_back(1.0);
    unsorted.push_back(5.0);
    unsorted.push_back(3.0);
    unsorted.push_back(2.0);
    unsorted.push_back(4.0);
    unsorted.push_back(10.0);

    sort_indices(unsorted, &indices, std::greater_equal<double>());

    return 0;
}
