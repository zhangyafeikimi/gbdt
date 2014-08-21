#include "lm-scorer.h"
#include "sample.h"
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <functional>

const double NDCGScorer::LOG2 = log((double)2.0);

NDCGScorer::NDCGScorer(size_t k)
    : k_(k)
{
    idcg_cache_.reserve(1000);
}

double NDCGScorer::gain(size_t label) const
{
    if (label >= gain_cache_.size())
    {
        size_t old_size = gain_cache_.size();
        size_t new_size = label + 1;
        gain_cache_.reserve(new_size);
        for (size_t i=old_size; i<new_size; i++)
            gain_cache_.push_back(pow(2.0, i) - 1.0);
    }

    return gain_cache_[label];
}

double NDCGScorer::discount(size_t index) const
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

double NDCGScorer::idcg(const std::vector<size_t>& labels, size_t qid, size_t top_k) const
{
    if (qid >= idcg_cache_.size())
    {
        assert(qid == idcg_cache_.size());
        double _idcg = idcg(labels, top_k);
        idcg_cache_.push_back(_idcg);
        return _idcg;
    }

    return idcg_cache_[qid];
}

double NDCGScorer::idcg(const std::vector<size_t>& labels, size_t top_k) const
{
    std::vector<size_t> sorted_labels = labels;
    double dcg = 0.0;
    std::sort(sorted_labels.begin(), sorted_labels.end(), std::greater<size_t>());
    for (size_t i=0; i<top_k; i++)
        dcg += gain(sorted_labels[i]) * discount(i);
    return dcg;
}

void NDCGScorer::get_delta_with_idcg(
    const std::vector<size_t>& labels,
    double _idcg,
    size_t top_k,
    SymmetricMatrixD * delta) const
{
    size_t size = labels.size();
    delta->resize(size);
    for (size_t i=0; i<top_k; i++)
    {
        for (size_t j=i+1; j<size; j++)
        {
            if (_idcg > 0.0)
            {
                double d = (gain(labels[i]) - gain(labels[j])) * (discount(i) - discount(j)) / _idcg;
                delta->at(i, j) = fabs(d);
            }
        }
    }
}

void NDCGScorer::get_delta(const std::vector<size_t>& labels, SymmetricMatrixD * delta) const
{
    size_t size = labels.size();
    size_t top_k = (size > k_) ? k_ : size;
    double _idcg = idcg(labels, top_k);
    get_delta_with_idcg(labels, _idcg, size, delta);
}

void NDCGScorer::get_delta(const std::vector<size_t>& labels, size_t qid, SymmetricMatrixD * delta) const
{
    size_t size = labels.size();
    size_t top_k = (size > k_) ? k_ : size;
    double _idcg = idcg(labels, qid, top_k);
    get_delta_with_idcg(labels, _idcg, size, delta);
}

void NDCGScorer::get_score(const std::vector<size_t>& labels,
                           double * ndcg,
                           double * dcg,
                           double * _idcg) const
{
    size_t size = labels.size();
    size_t top_k = (size > k_) ? k_ : size;
    *_idcg = idcg(labels, top_k);

    double _dcg = 0.0;
    for (size_t i=0; i<top_k; i++)
        _dcg += gain(labels[i]) * discount(i);

    if (_dcg < EPS && *_idcg < EPS)
    {
        *ndcg = 0.0;
        *dcg = _dcg;
    }
    else
    {
        *ndcg = _dcg / *_idcg;
        *dcg = _dcg;
    }
}
