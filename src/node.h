#ifndef GBDT_NODE_H
#define GBDT_NODE_H

#include "param.h"
#include "sample.h"
#include <assert.h>
#include <stdio.h>
#include <algorithm>

template <class T>
class TreeNodeBase
{
private:
    const TreeParam& param_;
    const size_t level_;

    T * left_;
    T * right_;
    XYSetRef set_;
    // loss of current tree and all preceding trees
    double total_loss_;
    // loss of current split
    double loss_;

    // inner node only
    // split position information
    size_t split_x_index_;
    kXType split_x_type_;
    CompoundValue split_x_value_;

    // leaf node only
    bool leaf_;
    // predicted y in this node
    double y_;

protected:
    // pseudo response
    std::vector<double> response_;
    // F(x) for only training samples in this tree node.
    // NOTE 1: even for a root node,
    // it is not the F(x) for the whole tree,
    // but only the root node.
    // NOTE 2: F(x) is the sums of predicted y of all preceding trees.
    std::vector<double> fx_;

protected:
    TreeNodeBase(const TreeParam& param, size_t level)
        : param_(param), level_(level),
        total_loss_(0.0), loss_(0.0) {}

public:
    const TreeParam& param() const {return param_;}
    size_t level() const {return level_;}

    T *& left() {return left_;}
    const T * left() const {return left_;}
    T *& right() {return right_;}
    const T * right() const {return right_;}
    XYSetRef& set() {return set_;}
    const XYSetRef& set() const {return set_;}
    double& total_loss() {return total_loss_;}
    double total_loss() const {return total_loss_;}
    double& loss() {return loss_;}
    double loss() const {return loss_;}

    size_t& split_x_index() {return split_x_index_;}
    size_t split_x_index() const {return split_x_index_;}
    kXType& split_x_type() {return split_x_type_;}
    bool split_is_numerical() const {return split_x_type_ == kXType_Numerical;}
    CompoundValue& split_x_value() {return split_x_value_;}
    double split_get_double() const {return split_x_value_.d();}
    int split_get_int() const {return split_x_value_.i();}

    bool& leaf() {return leaf_;}
    bool is_leaf() const {return leaf_;}
    double& y() {return y_;}
    double y() const {return y_;}

    virtual ~TreeNodeBase()
    {
        if (left_)
            delete left_;
        if (right_)
            delete right_;
    }

    // get some unique x values(get most 'max_size' x values)
    // for finding the 'best' x split
    // TODO: if learning rate is 1.0,
    // we can get a pre-sorted list of unique x values
    void get_unique_x_values(
        size_t x_index,
        size_t max_size,
        CompoundValueVector * x_values) const
    {
        x_values->clear();
        for (size_t i=0, s=std::min(max_size, set().size()); i<s; i++)
            x_values->push_back(set().get(i).x(x_index));

        kXType xtype = set().get_xtype(x_index);
        if (xtype == kXType_Numerical)
        {
            std::sort(x_values->begin(), x_values->end(), CompoundValueDoubleLess());
        }
        else
        {
            std::sort(x_values->begin(), x_values->end(), CompoundValueIntLess());
            x_values->erase(std::unique(x_values->begin(), x_values->end(), CompoundValueIntEqual()),
                x_values->end());
        }
    }

    double predict(const CompoundValueVector& X) const
    {
        return __predict(this, X);
    }

    void drain()
    {
        set().clear();
        response_.clear();
        fx_.clear();
        if (left())
            left()->drain();
        if (right())
            right()->drain();
    }

    // for root
    void add_xy_set_fxs(const XYSet& full_set, const std::vector<double>& full_fx)
    {
        set().load(full_set);
        fx_ = full_fx;
    }

    // for non-root
    void add_xy_fx(const XY& xy, double f)
    {
        set().add(xy);
        fx_.push_back(f);
    }

private:
    static double __predict(const TreeNodeBase * node, const CompoundValueVector& X)
    {
        for (;;)
        {
            if (node->is_leaf())
                return node->y();

            if (node->split_is_numerical())
            {
                if (X[node->split_x_index()].d() <= node->split_get_double())
                    node = node->left();
                else
                    node = node->right();
            }
            else
            {
                if (X[node->split_x_index()].i() == node->split_get_int())
                    node = node->left();
                else
                    node = node->right();
            }
            assert(node);
        }
    }
};

#endif// GBDT_NODE_H
