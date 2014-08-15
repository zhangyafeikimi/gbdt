#include "lm.h"
#include "json.h"
#include "node.h"
#include <assert.h>

class LambdaMARTNode : public TreeNodeBase
{
private:
    std::vector<double> weights_;
    // for root
    const std::vector<size_t> * n_samples_per_query_;

public:
    LambdaMARTNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level), n_samples_per_query_(0) {}

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
        //*y0 = weighted_median_y(full_set);
        //full_fx->assign(full_set.size(), *y0);
    }

protected:
    virtual void add_data(const XY& xy, const LambdaMARTNode * parent, size_t _index)
    {
        assert(!is_root());
        set().add(xy);

        response_.push_back(parent->response_[_index]);
        weights_.push_back(parent->weights_[_index]);
    }

    virtual void clear()
    {
        set().clear();
        response_.clear();
        weights_.clear();
    }

    virtual void update_response(const std::vector<double>& fx)
    {
        assert(param().gbdt_sample_rate >= 1.0);
        assert(n_samples_per_query_);
        //assert(response_.empty());
        //const XYSetRef& xy_set = set();
        //assert(xy_set.size() == fx.size());
        //for (size_t i=0, s=fx.size(); i<s; i++)
        //    response_.push_back(sign((xy_set.get(i).y() - fx[i])));
    }

    virtual void update_predicted_y()
    {
        //const XYSetRef& xy_set = set();
        //std::vector<XW> response_weight;
        //for (size_t i=0, s=xy_set.size(); i<s; i++)
        //{
        //    const XY& xy = xy_set.get(i);
        //    response_weight.push_back(XW(response_[i], xy.weight()));
        //}
        //// readjust leaf values by the weighted median values
        //y() = weighted_median(&response_weight);
    }
};

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

LambdaMARTTrainer::LambdaMARTTrainer(const XYSet& set, const TreeParam& param)
    : full_set_(set), param_(param), full_fx_()
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
        TreeNodeBase * tree = holder_->train(full_set_, param_, &full_fx_);
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
