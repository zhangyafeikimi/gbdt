#ifndef GBDT_LAMBDA_MART_SCORER_H
#define GBDT_LAMBDA_MART_SCORER_H

#include "sample.h"
#include "lm-util.h"

template <class T, class ScoreOfT>
class NDCGScorer
{
private:
    const size_t k_;
    const ScoreOfT score_of_t_;
public:
    NDCGScorer(size_t k)
        : k_(k), score_of_t_() {}

    void getDelta(const T * sequence, size_t size, SymmetricMatrixD * delta)
    {

    }
};

#endif// GBDT_LAMBDA_MART_SCORER_H
