#include "gain.h"
#include "node.h"
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <list>

static double entropy_part(double n, double p)
{
    static double log2 = log(2);
    if (n < EPS)
        return 0.0;
    else
        return - p * log(p) / log2;
}

static double entropy(double positive, double negative)
{
    double total = positive + negative;
    if (total < EPS)
        return 0.0;

    double pos_p = positive / total;
    double neg_p = negative / total;
    return entropy_part(positive, pos_p) + entropy_part(negative, neg_p);
}

static double information_gain(double positive, double negative,
                               double le_positive, double le_negative,
                               double gt_positive, double gt_negative)
{
    double n = positive + negative;
    double le_n = le_positive + le_negative;
    double gt_n = gt_positive + gt_negative;
    return entropy(positive, negative)
        - le_n / n * entropy(le_positive, le_negative)
        - gt_n / n * entropy(gt_positive, gt_negative);
}

class TreeGainNode : public TreeNode<TreeGainNode>
{
private:
    TreeGainNode(const TreeParam& param, size_t level)
        : TreeNode<TreeGainNode>(param, level) {}

    void information_gain_all(
        size_t * _split_x_index,
        kXType * _split_x_type,
        CompoundValue * _split_x_value,
        double * max_gain) const
    {
        *max_gain = -1.0;
        for (size_t x_index=0, s=set().get_xtype_size(); x_index<s; x_index++)
        {
            kXType x_type = set().get_xtype(x_index);
            CompoundValue x_value;
            double gain;
            information_gain_X(x_index, x_type, &x_value, &gain);
            if (gain > *max_gain)
            {
                *_split_x_index = x_index;
                *_split_x_type = x_type;
                *_split_x_value = x_value;
                *max_gain = gain;
            }
        }
    }

    void information_gain_X(
        size_t _split_x_index,
        kXType _split_x_type,
        CompoundValue * _split_x_value,
        double * max_gain) const
    {
        CompoundValueVector x_values;
        get_unique_x_values(_split_x_index, param().max_x_values_number, &x_values);
        *max_gain = -1.0;
        for (size_t i=0, s=x_values.size(); i<s; i++)
        {
            CompoundValue x_value = x_values[i];
            double gain;
            information_gain_x(_split_x_index, _split_x_type, x_value, &gain);
            if (gain > *max_gain)
            {
                *_split_x_value = x_value;
                *max_gain = gain;
            }
        }
    }

    void information_gain_x(
        size_t _split_x_index,
        kXType _split_x_type,
        CompoundValue _split_x_value,
        double * gain) const
    {
        if (_split_x_type == kXType_Numerical)
            information_gain_x_numerical(_split_x_index, _split_x_value.d(), gain);
        else
            information_gain_x_category(_split_x_index, _split_x_value.i(), gain);
    }

    void information_gain_x_numerical(
        size_t _split_x_index,
        double _split_x_value,
        double * gain) const
    {
        double split_x_value_double = _split_x_value;
        double le_positive = 0.0;// <= _split_x_value
        double le_negative = 0.0;// <= _split_x_value
        double gt_positive = 0.0;// > _split_x_value
        double gt_negative = 0.0;// > _split_x_value
        for (size_t i=0, s=set().size(); i<s; i++)
        {
            const XY& xy = set().get(i);
            double weight = xy.weight();
            double x = xy.x(_split_x_index).d();
            if (x <= split_x_value_double)
            {
                if (xy.y() < EPS)
                    le_negative += weight;
                else
                    le_positive += weight;
            }
            else
            {
                if (xy.y() < EPS)
                    gt_negative += weight;
                else
                    gt_positive += weight;
            }
        }
        *gain = information_gain(set().positive(), set().negative(),
            le_positive, le_negative, gt_positive, gt_negative);
    }

    void information_gain_x_category(
        size_t _split_x_index,
        int _split_x_value,
        double * gain) const
    {
        int split_x_value_int = _split_x_value;
        double le_positive = 0.0;// == _split_x_value
        double le_negative = 0.0;// == _split_x_value
        double gt_positive = 0.0;// != _split_x_value
        double gt_negative = 0.0;// != _split_x_value
        for (size_t i=0, s=set().size(); i<s; i++)
        {
            const XY& xy = set().get(i);
            double weight = xy.weight();
            int x = xy.x(_split_x_index).i();
            if (x == split_x_value_int)
            {
                if (xy.y() < EPS)
                    le_negative += weight;
                else
                    le_positive += weight;
            }
            else
            {
                if (xy.y() < EPS)
                    gt_negative += weight;
                else
                    gt_positive += weight;
            }
        }
        *gain = information_gain(set().positive(), set().negative(),
            le_positive, le_negative, gt_positive, gt_negative);
    }

public:
    static TreeGainNode * create_root(const XYSet& set, const TreeParam& param)
    {
        TreeGainNode * root = new TreeGainNode(param, 0);
        root->left() = 0;
        root->right() = 0;
        root->set().load(set);
        root->leaf() = false;
        assert(root->set().get_xtype_size() != 0);
        assert(root->set().size() != 0);
        return root;
    }

    static void split(TreeGainNode * pparent)
    {
        double max_gain;
        pparent->information_gain_all(&pparent->split_x_index(),
            &pparent->split_x_type(),
            &pparent->split_x_value(),
            &max_gain);

        XYSetRef& parent_xy_set = pparent->set();

        TreeGainNode * pleft = new TreeGainNode(pparent->param(), pparent->level() + 1);
        pleft->left() = 0;
        pleft->right() = 0;
        pleft->set().spec() = parent_xy_set.spec();
        pleft->leaf() = false;

        TreeGainNode * pright = new TreeGainNode(pparent->param(), pparent->level() + 1);
        pright->left() = 0;
        pright->right() = 0;
        pright->set().spec() = parent_xy_set.spec();
        pright->leaf() = false;

        pparent->left() = pleft;
        pparent->right() = pright;

        size_t _split_x_index = pparent->split_x_index();

        if (pparent->split_is_numerical())
        {
            double split_x_value_double = pparent->split_get_double();
            for (size_t i=0, s=parent_xy_set.size(); i<s; i++)
            {
                const XY& xy = parent_xy_set.get(i);
                double x = xy.x(_split_x_index).d();
                if (x <= split_x_value_double)
                    pleft->set().add(xy);
                else
                    pright->set().add(xy);
            }
        }
        else
        {
            double split_x_value_int = pparent->split_get_int();
            for (size_t i=0, s=parent_xy_set.size(); i<s; i++)
            {
                const XY& xy = parent_xy_set.get(i);
                int x = xy.x(_split_x_index).i();
                if (x == split_x_value_int)
                    pleft->set().add(xy);
                else
                    pright->set().add(xy);
            }
        }

        parent_xy_set.clear();
    }
};

TreeGain::TreeGain(const XYSet& set, const TreeParam& param)
    : set_(set), param_(param), root_(0) {}

TreeGain::~TreeGain()
{
    if (root_)
        delete root_;
}

void TreeGain::train()
{
    assert(root_ == 0);
    root_ = TreeGainNode::create_root(set_, param_);
    std::list<TreeGainNode *> stack;
    stack.push_back(root_);
    size_t leaf_size = 0;

    while (!stack.empty())
    {
        TreeGainNode * node = stack.back();
        stack.pop_back();

        size_t level = node->level();
        double positive = node->set().positive();
        double negative = node->set().negative();
        double total = positive + negative;
        if (level >= param_.max_level
            || leaf_size >= param_.max_leaf_number
            || std::max(positive, negative) / total >= param_.leaf_threshold
            || node->set().size() <= param_.min_values_in_leaf)
        {
            node->leaf() = true;
            node->y() = positive / total;
            leaf_size++;
            continue;
        }

        TreeGainNode::split(node);
        stack.push_back(node->left());
        stack.push_back(node->right());
    }
}

double TreeGain::predict(const CompoundValueVector& X) const
{
    assert(root_);
    return root_->predict(X);
}

void TreeGain::dump_dot(FILE * fp) const
{
    root_->dump_dot(fp);
}
