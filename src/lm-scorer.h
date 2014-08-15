#ifndef GBDT_LAMBDA_MART_SCORER_H
#define GBDT_LAMBDA_MART_SCORER_H

#include "sample.h"
#include "lm-util.h"
#include <math.h>
#include <algorithm>
#include <functional>
#include <vector>

static double LOG2 = log((double)2.0);

// NOTE: We assume labels are integers and 2^label can't be too large
// (namely, the range of size_t).
// Actually these two assumptions are facts in LECTOR 4.0.
// So we want LabelOfT to return an integer, although labels in memory are double.
template <class T, class LabelOfT>
class NDCGScorer
{
private:
    const size_t k_;
    const LabelOfT label_of_t_;

    static std::vector<double> gain_cache_;// caches 2^{label} - 1
    static std::vector<double> discount_cache_;// caches 1/\log_2(i+2), i=0,1,...

private:
    double gain(size_t label) const
    {
        if (label >= gain_cache_.size())
        {
            size_t old_size = gain_cache_.size();
            size_t new_size = label + 1;
            gain_cache_.reserve(new_size);
            for (size_t i=old_size; i<new_size; i++)
                gain_cache_.push_back((1 << label) - 1.0);
        }

        return gain_cache_[label];
    }

    double discount(size_t index) const
    {
        if (index >= discount_cache_.size())
        {
            size_t new_size = discount_cache_.size() + 1000;
            while (new_size <= index)
                new_size += 1000;
            discount_cache_.reserve(new_size);
            for (size_t i=discount_cache_.size(); i<new_size; i++)
                discount_cache_.push_back(1.0/(log((double)(i+2))/LOG2));
        }

        return discount_cache_[index];
    }

    double get_ideal_dcg(const std::vector<size_t>& labels, size_t top_k)
    {
        std::vector<size_t> sorted_labels = *labels;
        double dcg = 0.0;
        std::sort(sorted_labels.begin(), sorted_labels.end(), std::greater_equal<size_t>());
        for (size_t i=0; i<top_k; i++)
            dcg += gain(sorted_labels[i]) * discount(i);
        return dcg;
    }

public:
    NDCGScorer(size_t k)
        : k_(k), label_of_t_() {}

    void get_delta(const T * sequence, size_t size, SymmetricMatrixD * delta)
    {
        size_t top_k = (size > k_) ? k_ : size;
        std::vector<size_t> labels;

        for (size_t i=0; i<size; i++)
            labels.push_back(label_of_t_(sequence[i]));

        double ideal_dcg = get_ideal_dcg(labels, top_k);
        for (size_t i=0; i<top_k; i++)
        {
            for (size_t j=i+1; j<size; j++)
            {
                if (ideal_dcg > 0)
                {
                    delta->at(i, j) = abs(
                        (gain(labels[i]) - gain(labels[j])) * (discount(i) - discount(j)) / ideal_dcg
                        );
                }
            }
        }
    }
};

#endif// GBDT_LAMBDA_MART_SCORER_H
