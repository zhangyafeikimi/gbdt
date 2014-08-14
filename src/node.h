#ifndef GBDT_NODE_H
#define GBDT_NODE_H

#include "param.h"
#include "sample.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <list>

#if defined USE_10000_RANDOM
// If we want to get a deterministic sequence of random number,
// turn on macro "USE_10000_RANDOM".
class Rand01
{
private:
    static float data_[10000];
    const float rate_;
    size_t i_;

public:
    explicit Rand01(double rate)
        : rate_((float)rate), i_(0) {}

    bool is_one()
    {
        i_ = (i_+1) % 10000;
        return data_[i_] < rate_;
    }
};

float Rand01::data_[10000] = {
#include "10000float.inl"
};
#else
class Rand01
{
private:
    const int threshold_1_;
    unsigned seed_;

public:
    explicit Rand01(double rate_1)
        : threshold_1_((int)(rate_1 * RAND_MAX)), seed_(0) {}

    bool is_one()
    {
#if defined _WIN32
        int i = rand();
#else
        int i = rand_r(&seed_);
#endif
        return i < threshold_1_;
    }
};
#endif

#define X_LIES_LEFT(x, _split_x_value, _split_x_type) \
    (_split_x_type)?((x.d()) <= (_split_x_value.d())):((x.i()) == (_split_x_value.i()))

class TreeNodeBase
{
private:
    const TreeParam& param_;
    const size_t level_;

    TreeNodeBase * left_;
    TreeNodeBase * right_;
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
    // predicted y in this leaf node
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
        left_(0), right_(0),
        total_loss_(0.0), loss_(0.0) {}

public:
    const TreeParam& param() const {return param_;}
    size_t level() const {return level_;}
    TreeNodeBase *& left() {return left_;}
    const TreeNodeBase * left() const {return left_;}
    TreeNodeBase *& right() {return right_;}
    const TreeNodeBase * right() const {return right_;}
    XYSetRef& set() {return set_;}
    const XYSetRef& set() const {return set_;}
    double& total_loss() {return total_loss_;}
    double total_loss() const {return total_loss_;}
    double& loss() {return loss_;}
    double loss() const {return loss_;}
    size_t& split_x_index() {return split_x_index_;}
    size_t split_x_index() const {return split_x_index_;}
    kXType& split_x_type() {return split_x_type_;}
    kXType split_x_type() const {return split_x_type_;}
    CompoundValue& split_x_value() {return split_x_value_;}
    const CompoundValue& split_x_value() const {return split_x_value_;}
    bool split_is_numerical() const {return split_x_type_ == kXType_Numerical;}
    double split_get_double() const {return split_x_value_.d();}
    int split_get_int() const {return split_x_value_.i();}
    bool& leaf() {return leaf_;}
    bool is_leaf() const {return leaf_;}
    double& y() {return y_;}
    double y() const {return y_;}

public:
    virtual ~TreeNodeBase()
    {
        if (left_)
            delete left_;
        if (right_)
            delete right_;
    }

    TreeNodeBase * train(
        const XYSet& full_set,
        const TreeParam& param,
        std::vector<double> * full_fx) const
    {
        assert(full_set.size() == full_fx->size());
        TreeNodeBase * root = create_root(full_set, param, *full_fx);
        root->build_tree();
        root->update_fx(full_set, full_fx);
        root->drain();
        return root;
    }

    double predict(const CompoundValueVector& X) const
    {
        return __predict(this, X);
    }

private:
    TreeNodeBase * create_root(
        const XYSet& full_set,
        const TreeParam& param,
        const std::vector<double>& full_fx) const
    {
        TreeNodeBase * root = clone(param, 0);
        root->leaf() = false;

        XYSetRef& root_xy_set = root->set();
        if (param.sample_rate >= 1.0)
        {
            root->add_xy_set_fxs(full_set, full_fx);
        }
        else
        {
            // sample 'full_set' and 'full_fx' together
            root_xy_set.spec() = &full_set.spec();
            root_xy_set.x_values() = &full_set.x_values();
            Rand01 r(param.sample_rate);
            for (size_t i=0, s=full_set.size(); i<s; i++)
            {
                if (r.is_one())
                    root->add_xy_fx(full_set.get(i), full_fx[i]);
            }
        }
        assert(root_xy_set.get_x_type_size() != 0);
        assert(root_xy_set.size() != 0);

        root->update_response();
        return root;
    }

    // for root
    inline void add_xy_set_fxs(const XYSet& full_set, const std::vector<double>& full_fx)
    {
        set().load(full_set);
        fx_ = full_fx;
    }

    // for root
    inline void add_xy_fx(const XY& xy, double f)
    {
        set().add(xy);
        fx_.push_back(f);
    }

    // for non-root
    inline void add_xy_response_fx(const XY& xy, double response, double f)
    {
        set().add(xy);
        response_.push_back(response);
        fx_.push_back(f);
    }

    void build_tree()
    {
        const TreeParam& _param = param();
        std::list<TreeNodeBase *> stack;
        stack.push_back(this);
        size_t leaf_size = 0;

        while (!stack.empty())
        {
            TreeNodeBase * node = stack.back();
            stack.pop_back();

            size_t level = node->level();
            if (level >= _param.max_level
                || leaf_size >= _param.max_leaf_number
                || node->set().size() <= _param.min_values_in_leaf)
            {
                node->leaf() = true;
                node->update_predicted_y();
                node->shrink();
                leaf_size++;
                continue;
            }

            split(node);
            stack.push_back(node->left());
            stack.push_back(node->right());
        }
    }

    void split(TreeNodeBase * parent) const
    {
        XYSetRef& parent_xy_set = parent->set();
        assert(parent_xy_set.size() != 0);
        double y_left = 0.0;
        double y_right = 0.0;
        parent->min_loss_on_all_features(&parent->split_x_index(),
            &parent->split_x_type(),
            &parent->split_x_value(),
            &y_left,
            &y_right,
            &parent->loss());

        TreeNodeBase * _left = clone(parent->param(), parent->level() + 1);
        _left->set().spec() = parent_xy_set.spec();
        _left->set().x_values() = parent_xy_set.x_values();
        _left->leaf() = false;
        _left->y() = y_left;

        TreeNodeBase * _right = clone(parent->param(), parent->level() + 1);
        _right->set().spec() = parent_xy_set.spec();
        _right->set().x_values() = parent_xy_set.x_values();
        _right->leaf() = false;
        _right->y() = y_right;

        parent->left() = _left;
        parent->right() = _right;

        size_t _split_x_index = parent->split_x_index();
        const CompoundValue& _split_x_value = parent->split_x_value();
        for (size_t i=0, s=parent_xy_set.size(); i<s; i++)
        {
            double response = parent->response_[i];
            double fx = parent->fx_[i];
            const XY& xy = parent_xy_set.get(i);
            const CompoundValue& x = xy.x(_split_x_index);
            if (X_LIES_LEFT(x, _split_x_value, parent->split_x_type()))
                _left->add_xy_response_fx(xy, response, fx);
            else
                _right->add_xy_response_fx(xy, response, fx);
        }

        assert(parent->set().size() ==
            _left->set().size() + _right->set().size());
    }

    void shrink()
    {
        if (param().learning_rate >= 1.0)
            return;
        y() = y() * param().learning_rate;
    }

    void update_fx(const XYSet& full_set, std::vector<double> * full_fx) const
    {
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            const XY& xy = full_set.get(i);
            (*full_fx)[i] += predict(xy.X());
        }
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

    void min_loss_on_all_features(
        size_t * _split_x_index,
        kXType * _split_x_type,
        CompoundValue * _split_x_value,
        double * _y_left,
        double * _y_right,
        double * min_loss) const
    {
        *min_loss = std::numeric_limits<double>::max();
        for (size_t x_index=0, s=set().get_x_type_size(); x_index<s; x_index++)
        {
            kXType x_type = set().get_x_type(x_index);
            CompoundValue x_value;
            double y_left = 0.0;
            double y_right = 0.0;
            double loss;
            min_loss_on_one_feature(x_index, x_type, &x_value, &y_left, &y_right, &loss);
            if (loss < *min_loss)
            {
                *_split_x_index = x_index;
                *_split_x_type = x_type;
                *_split_x_value = x_value;
                *_y_left = y_left;
                *_y_right = y_right;
                *min_loss = loss;
            }
        }
    }

    void min_loss_on_one_feature(
        size_t _split_x_index,
        kXType _split_x_type,
        CompoundValue * _split_x_value,
        double * _y_left,
        double * _y_right,
        double * min_loss) const
    {
        const CompoundValueVector& unique_x_values = set().get_x_values(_split_x_index);
        *min_loss = std::numeric_limits<double>::max();
        for (size_t i=0, s=unique_x_values.size(); i<s; i++)
        {
            const CompoundValue& x_value = unique_x_values[i];
            double y_left;
            double y_right;
            double loss;
            loss_x(_split_x_index, _split_x_type, x_value, &y_left, &y_right, &loss);
            if (loss < *min_loss)
            {
                *_split_x_value = x_value;
                *_y_left = y_left;
                *_y_right = y_right;
                *min_loss = loss;
            }
        }
    }

    void loss_x(
        size_t _split_x_index,
        kXType _split_x_type,
        const CompoundValue& _split_x_value,
        double * _y_left,
        double * _y_right,
        double * _loss) const
    {
        double n_left = 0.0;
        double n_right = 0.0;
        double y_left = 0.0;
        double y_right = 0.0;

        for (size_t i=0, s=set().size(); i<s; i++)
        {
            const XY& xy = set().get(i);
            const CompoundValue& x = xy.x(_split_x_index);
            double weight = xy.weight();
            double response = response_[i];
            if (X_LIES_LEFT(x, _split_x_value, _split_x_type))
            {
                y_left += response * weight;
                n_left += weight;
            }
            else
            {
                y_right += response * weight;
                n_right += weight;
            }
        }

        if (n_left > EPS)
            y_left /= n_left;
        if (n_right > EPS)
            y_right /= n_right;

        *_y_left = y_left;
        *_y_right = y_right;
        __loss_x(_split_x_index, _split_x_type, _split_x_value, y_left, y_right, _loss);
    }

    void __loss_x(
        size_t _split_x_index,
        kXType _split_x_type,
        const CompoundValue& _split_x_value,
        double _y_left,
        double _y_right,
        double * _loss) const
    {
        double ls_loss = 0.0;
        for (size_t i=0, s=set().size(); i<s; i++)
        {
            const XY& xy = set().get(i);
            const CompoundValue& x = xy.x(_split_x_index);
            double weight = xy.weight();
            double diff;
            if (X_LIES_LEFT(x, _split_x_value, _split_x_type))
                diff = response_[i] - _y_left;
            else
                diff = response_[i] - _y_right;
            // weighted square loss
            ls_loss += (diff * diff * weight);
        }
        *_loss = ls_loss;
    }

    static double __predict(const TreeNodeBase * node, const CompoundValueVector& X)
    {
        for (;;)
        {
            if (node->is_leaf())
                return node->y();

            const CompoundValue& x = X[node->split_x_index()];
            const CompoundValue& _split_x_value = node->split_x_value();
            if (X_LIES_LEFT(x, _split_x_value, node->split_x_type()))
                node = node->left();
            else
                node = node->right();
            assert(node);
        }
    }

public:
    virtual TreeNodeBase * clone(
        const TreeParam& param,
        size_t level) const = 0;
    virtual void initial_fx(
        const XYSet& full_set,
        std::vector<double> * full_fx,
        double * y0) const = 0;
    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const = 0;

protected:
    // for root
    virtual void update_response() = 0;
    virtual void update_predicted_y() = 0;
};

static const TreeParam EMPTY_PARAM;

class TreeNodePredictor : public TreeNodeBase
{
private:
    TreeNodePredictor(size_t level)
        : TreeNodeBase(EMPTY_PARAM, level) {}

public:
    static TreeNodePredictor * create()
    {
        return new TreeNodePredictor(0);
    }

    virtual TreeNodeBase * clone(
        const TreeParam& param,
        size_t level) const
    {assert(0);return 0;}
    virtual void initial_fx(
        const XYSet& full_set,
        std::vector<double> * full_fx,
        double * y0) const {assert(0);}
    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const
    {assert(0);return 0.0;}

protected:
    virtual void update_response() {assert(0);}
    virtual void update_predicted_y() {assert(0);}
};

#endif// GBDT_NODE_H
