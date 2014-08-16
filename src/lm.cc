#include "lm.h"
#include "lm-scorer.h"
#include "json.h"
#include "node.h"
#include <assert.h>
#include <math.h>

class LambdaMARTNode : public TreeNodeBase
{
private:
    std::vector<double> weights_;
    // for root
    const std::vector<size_t> * n_samples_per_query_;

    static double mean_y(const XYSet& full_set)
    {
        double total_y  = 0.0;
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            total_y += xy.y();
        }
        return total_y / full_set.size();
    }

    struct XYLabelGreater
    {
        bool operator()(const XY * a, const XY * b) const
        {
            return a->y() > b->y();
        }
    };

public:
    LambdaMARTNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level), n_samples_per_query_(0)
    {
        assert(param.gbdt_sample_rate >= 1.0);
    }

    LambdaMARTNode * train(
        const XYSet& full_set,
        const std::vector<size_t>& n_samples_per_query,
        const TreeParam& param,
        std::vector<double> * full_fx) const
    {
        LambdaMARTNode * root = clone(param, 0);
        root->n_samples_per_query_ = &n_samples_per_query;
        root->do_train(full_set, param, full_fx);
        return root;
    }

    virtual LambdaMARTNode * clone(
        const TreeParam& param,
        size_t level) const
    {
        return new LambdaMARTNode(param, level);
    }

    virtual void initial_fx(const XYSet& full_set,
        std::vector<double> * full_fx, double * y0) const
    {
        *y0 = mean_y(full_set);
        full_fx->assign(full_set.size(), *y0);
    }

protected:
    virtual void add_data(const XY& xy, const TreeNodeBase * parent, size_t _index)
    {
        assert(!is_root());
        set().add(xy);
        LambdaMARTNode * lm_parent = (LambdaMARTNode *)parent;
        response_.push_back(lm_parent->response_[_index]);
        weights_.push_back(lm_parent->weights_[_index]);
    }

    virtual void clear()
    {
        set().clear();
        response_.clear();
        weights_.clear();
    }

    virtual void update_response(const std::vector<double>& fx)
    {
        // update 'response_' and 'weights_' together
        assert(n_samples_per_query_);
        assert(response_.empty());
        assert(weights_.empty());

        const XYSetRef& xy_set = set();
        assert(xy_set.size() == fx.size());
        response_.resize(xy_set.size(), 0.0);
        weights_.resize(xy_set.size(), 0.0);

        size_t ndcg_k = param().lm_ndcg_k;
        const NDCGScorer ndcg_scorer(ndcg_k);

        size_t begin = 0;
        for (size_t i=0, s=n_samples_per_query_->size(); i<s; i++)
        {
            // for each query-result list
            const XY * const * results = &xy_set.sample()[begin];
            size_t result_size = (*n_samples_per_query_)[i];

            // sort 'results'
            std::vector<size_t> indices;
            sort_indices(results, result_size, &indices, XYLabelGreater());
            // now 'results[indices[j]]'

            SymmetricMatrixD delta;
            std::vector<size_t> labels;
            for (size_t j=0; j<result_size; j++)
                labels.push_back((size_t)results[indices[j]]->y());
            ndcg_scorer.get_delta(labels, &delta);

            // 'j', 'k' are indices in 'indices' and 'results[indices[j]]'.
            // 'jj', 'kk' are indices in 'xy_set', 'response_', 'weights_' and 'fx'.
            for (size_t j=0; j<result_size; j++)
            {
                // for each result in the sorted query-result list 'results[indices[j]]'
                size_t jj = indices[j] + begin;
                const XY * xy_j = results[indices[j]];
                for (size_t k=0; k<result_size; k++)
                {
                    if (j>ndcg_k && k>ndcg_k)
                        break;

                    size_t kk = indices[k] + begin;
                    const XY * xy_k = results[indices[k]];
                    if (xy_j->y() > xy_k->y())
                    {
                        double delta_jk = delta.at(j, k);
                        if (delta_jk > 0.0)
                        {
                            double rho = 1.0 / (1.0 + exp(fx[jj] - fx[kk]));
                            double lambda = rho * delta_jk;
                            double lambda_d = rho * (1.0 - rho) * delta_jk;
                            response_[jj] += lambda;
                            response_[kk] -= lambda;
                            weights_[jj] += lambda_d;
                            weights_[kk] += lambda_d;
                        }
                    }
                }
            }

            begin += result_size;
        }
        assert(begin == xy_set.size());
    }

    virtual void update_predicted_y()
    {
        const XYSetRef& xy_set = set();
        assert(xy_set.size() == response_.size());
        assert(response_.size() == weights_.size());

        double sum_response = 0.0;
        double sum_weight = 0.0;

        for (size_t i=0, s=xy_set.size(); i<s; i++)
        {
            sum_response += response_[i];
            sum_weight += weights_[i];
        }

        if (sum_response < EPS && sum_weight < EPS)
            y() = 0.0;
        else
            y() = sum_response / sum_weight;
    }
};

/************************************************************************/
/* LambdaMARTPredictor and LambdaMARTTrainer */
/************************************************************************/
double LambdaMARTPredictor::predict(const CompoundValueVector& X) const
{
    assert(!trees_.empty());
    double y = y0_;
    for (size_t i=0, s=trees_.size(); i<s; i++)
        y += trees_[i]->predict(X);
    return y;
}

void LambdaMARTPredictor::clear()
{
    for (size_t i=0, s=trees_.size(); i<s; i++)
        delete trees_[i];
    trees_.clear();
}

LambdaMARTTrainer::LambdaMARTTrainer(
    const XYSet& set,
    const std::vector<size_t>& n_samples_per_query,
    const TreeParam& param)
    : full_set_(set), n_samples_per_query_(n_samples_per_query), param_(param), full_fx_()
{
    holder_ = new LambdaMARTNode(param, 0);
}

void LambdaMARTTrainer::train()
{
    assert(trees_.empty());

    holder_->initial_fx(full_set_, &full_fx_, &y0_);

    for (size_t i=0; i<param_.tree_number; i++)
    {
        printf("training tree No.%d... ", (int)i);
        LambdaMARTNode * tree = holder_->train(full_set_, n_samples_per_query_, param_, &full_fx_);
        trees_.push_back(tree);
        printf("OK\n");
    }
}

int LambdaMARTPredictor::load_json(FILE * fp)
{
    return ::load_json(fp, &y0_, &trees_);
}

void LambdaMARTTrainer::save_json(FILE * fp) const
{
    return ::save_json(fp, full_set_.spec(), y0_, trees_);
}
