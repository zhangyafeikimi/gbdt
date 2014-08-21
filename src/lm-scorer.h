#ifndef GBDT_LAMBDA_MART_SCORER_H
#define GBDT_LAMBDA_MART_SCORER_H

#include <stddef.h>
#include <vector>

// T is always double
template <class T>
class SymmetricMatrix
{
private:
    std::vector<T> m_;
    size_t row_;

public:
    SymmetricMatrix()
        : row_(0) {}

    void resize(size_t row)
    {
        m_.resize((row + 1) * row / 2);
        row_ = row;
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

typedef SymmetricMatrix<double> SymmetricMatrixD;

class NDCGScorer
{
private:
    const size_t k_;
    static const double LOG2;
    mutable std::vector<double> gain_cache_;// caches 2^{label} - 1
    mutable std::vector<double> discount_cache_;// caches 1/\log_2(i+2), i=0,1,...
    mutable std::vector<double> idcg_cache_;// caches qid -> ideal dcg

    double gain(size_t label) const;
    double discount(size_t index) const;
    double idcg(const std::vector<size_t>& labels, size_t qid, size_t top_k) const;
    double idcg(const std::vector<size_t>& labels, size_t top_k) const;
    void get_delta_with_idcg(
        const std::vector<size_t>& labels,
        double idcg,
        size_t top_k,
        SymmetricMatrixD * delta) const;

public:
    explicit NDCGScorer(size_t k);
    size_t get_cutoff() const {return k_;}
    void get_delta(const std::vector<size_t>& labels, SymmetricMatrixD * delta) const;
    void get_delta(const std::vector<size_t>& labels, size_t qid, SymmetricMatrixD * delta) const;
    void get_score(const std::vector<size_t>& labels,
        double * ndcg,
        double * dcg,
        double * idcg) const;
};

// TODO implement other metrics to use param().lm_metric

#endif// GBDT_LAMBDA_MART_SCORER_H
