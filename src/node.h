#ifndef GBDT_NODE_H
#define GBDT_NODE_H

#include "param.h"
#include "sample.h"

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

public:
    const TreeParam& param() const {return param_;}
    size_t level() const {return level_;}
    bool is_root() const {return level_ == 0;}
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

protected:
    TreeNodeBase(const TreeParam& param, size_t level);

public:
    virtual ~TreeNodeBase();
    TreeNodeBase * train(
        const XYSet& full_set,
        const TreeParam& param,
        std::vector<double> * full_fx) const;
    double predict(const CompoundValueVector& X) const;

protected:
    void do_train(
        const XYSet& full_set,
        const TreeParam& param,
        std::vector<double> * full_fx);

private:
    void sample_and_update_response(
        const XYSet& full_set,
        const TreeParam& param,
        const std::vector<double>& full_fx);
    void build_tree();
    void split();
    TreeNodeBase * fork() const;
    void split_data(TreeNodeBase * _left, TreeNodeBase * _right) const;
    void shrink();
    void update_fx(const XYSet& full_set, std::vector<double> * full_fx) const;
    void clear_tree();
    void min_loss_on_all_features(
        size_t * _split_x_index,
        kXType * _split_x_type,
        CompoundValue * _split_x_value,
        double * _y_left,
        double * _y_right,
        double * min_loss) const;
    void min_loss_on_one_feature(
        size_t _split_x_index,
        kXType _split_x_type,
        CompoundValue * _split_x_value,
        double * _y_left,
        double * _y_right,
        double * min_loss) const;
    void loss_x(
        size_t _split_x_index,
        kXType _split_x_type,
        const CompoundValue& _split_x_value,
        double * _y_left,
        double * _y_right,
        double * _loss) const;
    void __loss_x(
        size_t _split_x_index,
        kXType _split_x_type,
        const CompoundValue& _split_x_value,
        double _y_left,
        double _y_right,
        double * _loss) const;
    static double __predict(const TreeNodeBase * node, const CompoundValueVector& X);

public:
    virtual double total_loss(
        const XYSet& full_set,
        const std::vector<double>& full_fx) const;
    virtual TreeNodeBase * clone(
        const TreeParam& param,
        size_t level) const = 0;
    // for the first tree
    virtual void initial_fx(
        const XYSet& full_set,
        std::vector<double> * full_fx,
        double * y0) const = 0;

protected:
    virtual void add_data(const XY& xy, const TreeNodeBase * parent, size_t _index);
    virtual void clear();
    virtual void update_response(const std::vector<double>& fx) = 0;
    virtual void update_predicted_y() = 0;
};

class TreeNodePredictor : public TreeNodeBase
{
private:
    static const TreeParam EMPTY_PARAM;
    TreeNodePredictor(size_t level);
public:
    static TreeNodePredictor * create();
    virtual TreeNodeBase * clone(
        const TreeParam& param,
        size_t level) const;
    virtual void initial_fx(
        const XYSet& full_set,
        std::vector<double> * full_fx,
        double * y0) const;
protected:
    virtual void update_response(const std::vector<double>& fx);
    virtual void update_predicted_y();
};

#endif// GBDT_NODE_H
