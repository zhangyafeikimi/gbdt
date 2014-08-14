#ifndef GBDT_LAMBDA_MART_NODE_H
#define GBDT_LAMBDA_MART_NODE_H

#include "node.h"

class LambdaMARTNode : public TreeNodeBase
{
private:
    std::vector<double> weights_;

public:
    LambdaMARTNode(const TreeParam& param, size_t level)
        : TreeNodeBase(param, level) {}

    virtual TreeNodeBase * clone(
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

    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const
    {
        //assert(full_set.size() == full_fx.size());
        return 0.0;
    }

protected:
    virtual void add_data(const XY& xy, const TreeNodeBase * parent, size_t _index)
    {
        assert(!is_root());
        set().add(xy);

        LambdaMARTNode * p = (LambdaMARTNode *)parent;
        response_.push_back(p->response_[_index]);
        weights_.push_back(p->weights_[_index]);
    }

    virtual void clear()
    {
        set().clear();
        response_.clear();
        weights_.clear();
    }

    virtual void update_response(const std::vector<double>& fx)
    {
        //assert(response_.empty());
        //const XYSetRef& xy_set = set();
        //assert(xy_set.size() == fx.size());
        //for (size_t i=0, s=fx.size(); i<s; i++)
        //    response_.push_back(sign((xy_set.get(i).y() - fx[i])));
    }

    virtual void update_predicted_y()
    {
        const XYSetRef& xy_set = set();
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

#endif// GBDT_LAMBDA_MART_NODE_H
