#include "node.h"
#include <assert.h>
#include <stdlib.h>
#include <limits>
#include <list>

#if defined ENABLE_10000_RANDOM
// If we want to get a deterministic sequence of random number,
// turn on macro "ENABLE_10000_RANDOM".
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

TreeNodeBase::TreeNodeBase(const TreeParam& param, size_t level)
    : param_(param), level_(level),
    left_(0), right_(0),
    total_loss_(0.0), loss_(0.0) {}

TreeNodeBase::~TreeNodeBase()
{
    if (left_)
        delete left_;
    if (right_)
        delete right_;
}

TreeNodeBase * TreeNodeBase::train(
    const XYSet& full_set,
    const TreeParam& param,
    std::vector<double> * full_fx) const
{
    TreeNodeBase * root = clone(param, 0);
    root->do_train(full_set, param, full_fx);
    return root;
}

double TreeNodeBase::predict(const CompoundValueVector& X) const
{
    return __predict(this, X);
}

void TreeNodeBase::do_train(
    const XYSet& full_set,
    const TreeParam& param,
    std::vector<double> * full_fx)
{
    assert(full_set.size() == full_fx->size());
    leaf() = false;
    sample_and_update_response(full_set, param, *full_fx);
    build_tree();
    update_fx(full_set, full_fx);
    clear_tree();
}

void TreeNodeBase::sample_and_update_response(
    const XYSet& full_set,
    const TreeParam& param,
    const std::vector<double>& full_fx)
{
    assert(is_root());
    XYSetRef& xy_set = set();
    if (param.gbdt_sample_rate >= 1.0)
    {
        xy_set.load(full_set);
        update_response(full_fx);
    }
    else
    {
        std::vector<double> sampled_fx;
        // sample 'full_set' and 'full_fx' together
        xy_set.spec() = &full_set.spec();
        xy_set.x_values() = &full_set.x_values();
        Rand01 r(param.gbdt_sample_rate);
        for (size_t i=0, s=full_set.size(); i<s; i++)
        {
            if (r.is_one())
            {
                xy_set.add(full_set.get(i));
                sampled_fx.push_back(full_fx[i]);
            }
        }
        update_response(sampled_fx);
    }

    assert(xy_set.get_x_type_size() != 0);
    assert(xy_set.size() != 0);
}

void TreeNodeBase::build_tree()
{
    assert(is_root());
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

        node->split();
        stack.push_back(node->left());
        stack.push_back(node->right());
    }
}

void TreeNodeBase::split()
{
    const XYSetRef& xy_set = set();
    assert(xy_set.size() != 0);
    double y_left = 0.0;
    double y_right = 0.0;
    min_loss_on_all_features(&split_x_index(),
        &split_x_type(),
        &split_x_value(),
        &y_left,
        &y_right,
        &loss());

    TreeNodeBase * _left = fork();
    TreeNodeBase * _right = fork();
    left() = _left;
    right() = _right;
    _left->y() = y_left;
    _right->y() = y_right;
    split_data(_left, _right);
}

TreeNodeBase * TreeNodeBase::fork() const
{
    const XYSetRef& xy_set = set();
    TreeNodeBase * child = clone(param(), level() + 1);
    child->set().spec() = xy_set.spec();
    child->set().x_values() = xy_set.x_values();
    child->leaf() = false;
    return child;
}

void TreeNodeBase::split_data(TreeNodeBase * _left, TreeNodeBase * _right) const
{
    const XYSetRef& xy_set = set();
    size_t _split_x_index = split_x_index();
    const CompoundValue& _split_x_value = split_x_value();
    for (size_t i=0, s=xy_set.size(); i<s; i++)
    {
        const XY& xy = xy_set.get(i);
        const CompoundValue& x = xy.x(_split_x_index);
        if (X_LIES_LEFT(x, _split_x_value, split_x_type()))
            _left->add_data(xy, this, i);
        else
            _right->add_data(xy, this, i);
    }

    assert(xy_set.size() == _left->set().size() + _right->set().size());
}

void TreeNodeBase::shrink()
{
    if (param().learning_rate >= 1.0)
        return;
    y() = y() * param().learning_rate;
}

void TreeNodeBase::update_fx(const XYSet& full_set, std::vector<double> * full_fx) const
{
    assert(is_root());
    for (size_t i=0, s=full_set.size(); i<s; i++)
    {
        const XY& xy = full_set.get(i);
        (*full_fx)[i] += predict(xy.X());
    }
}

void TreeNodeBase::clear_tree()
{
    clear();
    if (left())
        left()->clear_tree();
    if (right())
        right()->clear_tree();
}

void TreeNodeBase::min_loss_on_all_features(
    size_t * _split_x_index,
    kXType * _split_x_type,
    CompoundValue * _split_x_value,
    double * _y_left,
    double * _y_right,
    double * min_loss) const
{
    const XYSetRef& xy_set = set();
    *min_loss = std::numeric_limits<double>::max();
    for (size_t x_index=0, s=xy_set.get_x_type_size(); x_index<s; x_index++)
    {
        kXType x_type = xy_set.get_x_type(x_index);
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

void TreeNodeBase::min_loss_on_one_feature(
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

void TreeNodeBase::loss_x(
    size_t _split_x_index,
    kXType _split_x_type,
    const CompoundValue& _split_x_value,
    double * _y_left,
    double * _y_right,
    double * _loss) const
{
    const XYSetRef& xy_set = set();
    double n_left = 0.0;
    double n_right = 0.0;
    double y_left = 0.0;
    double y_right = 0.0;

    for (size_t i=0, s=xy_set.size(); i<s; i++)
    {
        const XY& xy = xy_set.get(i);
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

    if (y_left < EPS && n_left < EPS)
        y_left = 0.0;
    else
        y_left /= n_left;

    if (y_right < EPS && n_right < EPS)
        y_right = 0.0;
    else
        y_right /= n_right;

    *_y_left = y_left;
    *_y_right = y_right;
    __loss_x(_split_x_index, _split_x_type, _split_x_value, y_left, y_right, _loss);
}

void TreeNodeBase::__loss_x(
    size_t _split_x_index,
    kXType _split_x_type,
    const CompoundValue& _split_x_value,
    double _y_left,
    double _y_right,
    double * _loss) const
{
    const XYSetRef& xy_set = set();
    double ls_loss = 0.0;
    for (size_t i=0, s=xy_set.size(); i<s; i++)
    {
        const XY& xy = xy_set.get(i);
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

double TreeNodeBase::__predict(const TreeNodeBase * node, const CompoundValueVector& X)
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

double TreeNodeBase::total_loss(
    const XYSet& full_set,
    const std::vector<double>& full_fx) const
{
    return 0.0;
}

void TreeNodeBase::add_data(const XY& xy, const TreeNodeBase * parent, size_t _index)
{
    assert(!is_root());
    set().add(xy);
    response_.push_back(parent->response_[_index]);
}

void TreeNodeBase::clear()
{
    set().clear();
    response_.clear();
}

/************************************************************************/
/* TreeNodePredictor */
/************************************************************************/
const TreeParam TreeNodePredictor::EMPTY_PARAM;

TreeNodePredictor::TreeNodePredictor(size_t level)
    : TreeNodeBase(EMPTY_PARAM, level) {}

TreeNodePredictor * TreeNodePredictor::create()
{
    return new TreeNodePredictor(0);
}

TreeNodeBase * TreeNodePredictor::clone(
    const TreeParam& param,
    size_t level) const
{
    assert(0);
    return 0;
}

void TreeNodePredictor::initial_fx(
    const XYSet& full_set,
    std::vector<double> * full_fx,
    double * y0) const
{
    assert(0);
}

void TreeNodePredictor::update_response(const std::vector<double>& fx)
{
    assert(0);
}

void TreeNodePredictor::update_predicted_y()
{
    assert(0);
}
