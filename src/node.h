#ifndef GBDT_NODE_H
#define GBDT_NODE_H

#include "param.h"
#include "sample.h"
#include <assert.h>
#include <stdio.h>
#include <algorithm>

template <class T>
class TreeNode
{
private:
    const TreeParam& param_;
    const size_t level_;

    T * left_;
    T * right_;
    XYSetRef set_;

    // inner node only
    size_t split_x_index_;
    kXType split_x_type_;
    CompoundValue split_x_value_;

    // leaf node only
    bool leaf_;
    double y_;

protected:
    TreeNode(const TreeParam& param, size_t level)
        : param_(param), level_(level) {}

public:
    const TreeParam& param() const {return param_;}
    size_t level() const {return level_;}

    T *& left() {return left_;}
    const T * left() const {return left_;}
    T *& right() {return right_;}
    const T * right() const {return right_;}
    XYSetRef& set() {return set_;}
    const XYSetRef& set() const {return set_;}

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

    virtual ~TreeNode()
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

    void dump_dot(FILE * fp) const
    {
        fprintf(fp, "digrapg G{\n");
        __dump_dot(fp);
        fprintf(fp, "}\n");
    }

    double predict(const CompoundValueVector& X) const
    {
        return __predict(this, X);
    }

private:
    void __dump_dot(FILE * fp) const
    {
        if (is_leaf())
        {
            fprintf(fp, "    node%p[label=\"leaf:value=%lf\"];\n", this, y());
        }
        else
        {
            fprintf(fp, "    node%p[label=\"non-leaf\"];\n", this);
            if (left())
                __dump_dot_child(fp, left());
            if (right())
                __dump_dot_child(fp, right());
        }
    }

    void __dump_dot_child(FILE * fp, const TreeNode * child) const
    {
        if (split_is_numerical())
            fprintf(fp, "    node%p -> node%p[label=\"if (X[%d] <= %lf)\"]\n",
            this, child, (int)split_x_index(), split_get_double());
        else
            fprintf(fp, "    node%p -> node%p[label=\"if (X[%d] <= %d)\"]\n",
            this, child, (int)split_x_index(), split_get_int());
        child->__dump_dot(fp);
    }

    static double __predict(const TreeNode * node, const CompoundValueVector& X)
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
