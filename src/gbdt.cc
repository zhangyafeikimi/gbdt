#include "gbdt.h"
#include "json.h"
#include "node.h"
#include <assert.h>
#include <math.h>
#include <algorithm>

static double weighted_mean_y(const XYSet& full_set)
{
    double total_y  = 0.0;
    double total_weight = 0.0;
    for (size_t i=0, s=full_set.size(); i<s; i++)
    {
        const XY& xy = full_set.get(i);
        double weight = xy.weight();
        total_y += xy.y() * weight;
        total_weight += weight;
    }
    return total_y / total_weight;
}

/************************************************************************/
/* LSLossNode */
/************************************************************************/
class LSLossNode : public TreeNodeBase
{
public:
    LSLossNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level) {}

    virtual LSLossNode * clone(
        const TreeParam& param,
        size_t level) const
    {
        return new LSLossNode(param, level);
    }

    virtual void initial_fx(
        const XYSet& full_set,
        std::vector<double> * full_fx,
        double * y0) const
    {
        *y0 = weighted_mean_y(full_set);
        full_fx->assign(full_set.size(), *y0);
    }

    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const
    {
        assert(full_set.size() == full_fx.size());
        double loss = 0.0;
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            double residual = full_set.get(i).y() - full_fx[i];
            loss += (residual * residual * xy.weight());
        }
        return loss;
    }

protected:
    virtual void update_response(const std::vector<double>& fx)
    {
        assert(response_.empty());
        const XYSetRef& xy_set = set();
        assert(xy_set.size() == fx.size());
        for (size_t i=0, s=fx.size(); i<s; i++)
            response_.push_back((xy_set.get(i).y() - fx[i]));
    }

    virtual void update_predicted_y() {}
};

/************************************************************************/
/* LADLossNode */
/************************************************************************/
class LADLossNode : public TreeNodeBase
{
private:
    static double sign(double y)
    {
        if (y >= 0.0)
            return 1.0;
        else
            return -1.0;
    }

    // x and its weight
    struct XW
    {
        double x;
        double w;
        XW(double _x, double _w) : x(_x), w(_w) {}
    };

    struct XWLess
    {
        bool operator()(const XW& a, const XW& b) const
        {
            return a.x < b.x;
        }
    };

    // http://stackoverflow.com/questions/9794558/weighted-median-computation
    static double weighted_median(std::vector<XW> * xw)
    {
        assert(!xw->empty());
        std::sort(xw->begin(), xw->end(), XWLess());
        double S = 0.0;
        for (size_t i=0, s=xw->size(); i<s; i++)
            S += (*xw)[i].w;
        size_t k = 0;
        double sum = S - (*xw)[0].w;

        while(sum > S/2.0)
        {
            ++k;
            sum -= (*xw)[k].w;
        }
        return (*xw)[k].x;
    }

    static double weighted_median_y(const XYSet& full_set)
    {
        std::vector<XW> xw;
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            xw.push_back(XW(xy.y(), xy.weight()));
        }
        return weighted_median(&xw);
    }

public:
    LADLossNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level) {}

    virtual LADLossNode * clone(
        const TreeParam& param,
        size_t level) const
    {
        return new LADLossNode(param, level);
    }

    virtual void initial_fx(const XYSet& full_set,
        std::vector<double> * full_fx, double * y0) const
    {
        *y0 = weighted_median_y(full_set);
        full_fx->assign(full_set.size(), *y0);
    }

    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const
    {
        assert(full_set.size() == full_fx.size());
        double loss = 0.0;
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            double residual = full_set.get(i).y() - full_fx[i];
            loss += fabs(residual) * xy.weight();
        }
        return loss;
    }

protected:
    virtual void update_response(const std::vector<double>& fx)
    {
        assert(response_.empty());
        const XYSetRef& xy_set = set();
        assert(xy_set.size() == fx.size());
        for (size_t i=0, s=fx.size(); i<s; i++)
            response_.push_back(sign((xy_set.get(i).y() - fx[i])));
    }

    virtual void update_predicted_y()
    {
        const XYSetRef& xy_set = set();
        if (xy_set.size() == 0)
        {
            y() = 0.0;
            return;
        }

        std::vector<XW> response_weight;
        for (size_t i=0, s=xy_set.size(); i<s; i++)
        {
            const XY& xy = xy_set.get(i);
            response_weight.push_back(XW(response_[i], xy.weight()));
        }
        // readjust leaf values by the weighted median values
        y() = weighted_median(&response_weight);
    }
};

/************************************************************************/
/* LogisticLossNode */
/************************************************************************/
class LogisticLossNode : public TreeNodeBase
{
public:
    LogisticLossNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level) {}

    virtual LogisticLossNode * clone(
        const TreeParam& param,
        size_t level) const
    {
        return new LogisticLossNode(param, level);
    }

    virtual void initial_fx(const XYSet& full_set,
        std::vector<double> * full_fx, double * y0) const
    {
        double _mean_y = weighted_mean_y(full_set);
        *y0 = 0.5 * log((1+_mean_y) / (1-_mean_y));
        full_fx->assign(full_set.size(), *y0);
    }

    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const
    {
        assert(full_set.size() == full_fx.size());
        double loss = 0.0;
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            loss += log(1 + exp(-2.0 * xy.y() * full_fx[i])) * xy.weight();
        }
        return loss;
    }

protected:
    virtual void update_response(const std::vector<double>& fx)
    {
        assert(response_.empty());
        const XYSetRef& xy_set = set();
        assert(xy_set.size() == fx.size());
        for (size_t i=0, s=fx.size(); i<s; i++)
        {
            double y = xy_set.get(i).y();
            double response = 2.0 * y / (1.0 + exp(2 * y * fx[i]));
            response_.push_back(response);
        }
    }

    virtual void update_predicted_y()
    {
        const XYSetRef& xy_set = set();
        if (xy_set.size() == 0)
        {
            y() = 0.0;
            return;
        }

        double numerator = 0.0, denominator = 0.0;
        for (size_t i=0, s=xy_set.size(); i<s; i++)
        {
            const XY& xy = xy_set.get(i);
            double weight = xy.weight();
            double response = response_[i];
            double abs_response = fabs(response);

            numerator += response * weight;
            denominator += abs_response * (2.0 - abs_response) * weight;
        }

        if (numerator < EPS && denominator < EPS)
            y() = 0.0;
        else
            y() = numerator / denominator;
    }
};

/************************************************************************/
/* GBDTPredictor and GBDTTrainer */
/************************************************************************/
double GBDTPredictor::predict(const CompoundValueVector& X) const
{
    assert(!trees_.empty());
    double y = y0_;
    for (size_t i=0, s=trees_.size(); i<s; i++)
        y += trees_[i]->predict(X);
    return y;
}

double GBDTPredictor::predict_logistic(const CompoundValueVector& X) const
{
    return 1.0 / (1.0 + exp(-2.0 * predict(X)));
}

void GBDTPredictor::clear()
{
    for (size_t i=0, s=trees_.size(); i<s; i++)
        delete trees_[i];
    trees_.clear();
}

GBDTTrainer::GBDTTrainer(const XYSet& set, const TreeParam& param)
    : full_set_(set), param_(param), full_fx_()
{
    if (param_.gbdt_loss == "lad")
    {
        holder_ = new LADLossNode(param, 0);
    }
    else if (param_.gbdt_loss == "logistic")
    {
        holder_ = new LogisticLossNode(param, 0);
    }
    else
    {
        holder_ = new LSLossNode(param, 0);
    }
}

GBDTTrainer::~GBDTTrainer()
{
    delete holder_;
}

double GBDTTrainer::total_loss() const
{
    return holder_->total_loss(full_set_, full_fx_);
}

static void record_loss_drop(const TreeNodeBase * node,
                             double current_loss,
                             std::vector<double> * loss_drop_vector)
{
    if (node->is_leaf())
        return;

    double& loss_drop = (*loss_drop_vector)[node->split_x_index()];
    double drop = current_loss - node->loss();
    loss_drop += drop;

    record_loss_drop(node->left(), node->loss(), loss_drop_vector);
    record_loss_drop(node->right(), node->loss(), loss_drop_vector);
}

void GBDTTrainer::dump_feature_importance() const
{
    if (param_.gbdt_sample_rate != 1.0)
        printf("sample rate is not 1.0, feature importance is unfair\n");

    std::vector<double> loss_drop_vector;
    loss_drop_vector.resize(full_set_.get_x_type_size(), 0.0);

    for (size_t i=0; i<trees_.size(); i++)
        record_loss_drop(trees_[i], trees_[i]->total_loss(), &loss_drop_vector);

    double total_drop = 0.0;
    for (size_t i=0; i<loss_drop_vector.size(); i++)
        total_drop += loss_drop_vector[i];
    for (size_t i=0; i<loss_drop_vector.size(); i++)
        printf("feature %d importance: %lf\n", (int)i, loss_drop_vector[i] / total_drop);
}

void GBDTTrainer::train()
{
    assert(trees_.empty());

    holder_->initial_fx(full_set_, &full_fx_, &y0_);
    if (param_.verbose)
        printf("total_loss=%lf\n", total_loss());

    for (size_t i=0; i<param_.tree_number; i++)
    {
        printf("training tree No.%d... ", (int)i);
        TreeNodeBase * tree = holder_->train(full_set_, param_, &full_fx_);
        trees_.push_back(tree);
        if (param_.verbose)
        {
            double _total_loss = total_loss();
            tree->total_loss() = _total_loss;
            printf("total_loss=%lf\n", _total_loss);
        }
        printf("OK\n");
    }

    if (param_.verbose)
        dump_feature_importance();
}

int GBDTPredictor::load_json(FILE * fp)
{
    return ::load_json(fp, &y0_, &trees_);
}

void GBDTTrainer::save_json(FILE * fp) const
{
    return ::save_json(fp, full_set_.spec(), y0_, trees_);
}
