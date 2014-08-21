#ifndef GBDT_TRAINING_SAMPLE_H
#define GBDT_TRAINING_SAMPLE_H

#include <stddef.h>
#include <vector>

#if !defined EPS
# define EPS (1e-9)
#endif

enum kXType
{
    kXType_Category = 0,
    kXType_Numerical = 1,
};

// training sample's feature specifications
class XYSpec
{
private:
    std::vector<kXType> x_types_;

public:
    size_t get_x_type_size() const {return x_types_.size();}
    kXType get_x_type(size_t i) const {return x_types_[i];}
    void add_x_type(kXType x_type) {x_types_.push_back(x_type);}
    void clear() {x_types_.clear();}
};

class CompoundValue
{
private:
    union
    {
        double d_;
        int i_;
        size_t label_;
    } value_;

public:
    CompoundValue() {value_.d_ = 0.0;}

    double& d() {return value_.d_;}
    double d() const {return value_.d_;}
    int& i() {return value_.i_;}
    int i() const {return value_.i_;}
    size_t& label() {return value_.label_;}
    size_t label() const {return value_.label_;}
};

typedef std::vector<CompoundValue> CompoundValueVector;

struct CompoundValueDoubleLess
{
    bool operator()(const CompoundValue& a, const CompoundValue& b) const
    {
        return a.d() < b.d();
    }
};

struct CompoundValueIntLess
{
    bool operator()(const CompoundValue& a, const CompoundValue& b) const
    {
        return a.i() < b.i();
    }
};

struct CompoundValueIntEqual
{
    bool operator()(const CompoundValue& a, const CompoundValue& b) const
    {
        return a.i() == b.i();
    }
};

// a training sample
class XY
{
private:
    CompoundValueVector X_;// X, features
    CompoundValue y_;// y or label
#if !defined DISABLE_WEIGHT
    double weight_;
#endif

public:
    size_t get_x_size() const {return X_.size();}
    CompoundValue& x(size_t i) {return X_[i];}
    const CompoundValue& x(size_t i) const {return X_[i];}
    const CompoundValueVector& X() const {return X_;}
    void add_x(const CompoundValue& _x) {X_.push_back(_x);}
    void resize_x(size_t s) {X_.resize(s);}

    double& y() {return y_.d();}
    double y() const {return y_.d();}

    size_t& label() {return y_.label();}
    size_t label() const {return y_.label();}

#if defined DISABLE_WEIGHT
    void set_weight(double weight) {}
    double weight() const {return 1.0;}
#else
    void set_weight(double weight) {weight_ = weight;}
    double weight() const {return weight_;}
#endif
};

struct XYLabelGreater
{
    bool operator()(const XY& a, const XY& b) const
    {
        return a.label() > b.label();
    }

    bool operator()(const XY * a, const XY * b) const
    {
        return a->label() > b->label();
    }
};

// a set of training samples
class XYSet
{
private:
    XYSpec spec_;
    std::vector<CompoundValueVector> x_values_;
    std::vector<XY> samples_;

public:
    XYSpec& spec() {return spec_;}
    const XYSpec& spec() const {return spec_;}

    std::vector<CompoundValueVector>& x_values() {return x_values_;}
    const std::vector<CompoundValueVector>& x_values() const {return x_values_;}

    std::vector<XY>& sample() {return samples_;}
    const std::vector<XY>& sample() const {return samples_;}

    size_t get_x_type_size() const {return spec_.get_x_type_size();}
    kXType get_x_type(size_t i) const {return spec_.get_x_type(i);}
    void add_x_type(kXType xtype) {spec_.add_x_type(xtype);}

    size_t get_x_values_size() const {return x_values_.size();}
    CompoundValueVector& get_x_values(size_t i) {return x_values_[i];}
    const CompoundValueVector& get_x_values(size_t i) const {return x_values_[i];}
    void add_x_values(const CompoundValueVector& x_values) {x_values_.push_back(x_values);}

    size_t size() const {return samples_.size();}
    XY& get(size_t i) {return samples_[i];}
    const XY& get(size_t i) const {return samples_[i];}
    void add(const XY& xy) {samples_.push_back(xy);}

    void clear()
    {
        spec_.clear();
        samples_.clear();
    }
};

// external reference to a set of training samples
class XYSetRef
{
private:
    // training sample specifications
    const XYSpec * spec_;
    // x_values_[i] is a collection of pre-sorted x values of the ith feature.
    // It is used when tree is being split.
    const std::vector<CompoundValueVector> * x_values_;
    // training samples
    std::vector<const XY *> samples_;

public:
    XYSetRef() {clear();}

    const XYSpec *& spec() {return spec_;}
    const XYSpec * spec() const {return spec_;}

    const std::vector<CompoundValueVector> *& x_values() {return x_values_;}
    const std::vector<CompoundValueVector> * x_values() const {return x_values_;}

    std::vector<const XY *>& sample() {return samples_;}
    const std::vector<const XY *>& sample() const {return samples_;}

    size_t get_x_type_size() const {return spec_->get_x_type_size();}
    kXType get_x_type(size_t i) const {return spec_->get_x_type(i);}

    size_t get_x_values_size() const {return x_values_->size();}
    const CompoundValueVector& get_x_values(size_t i) const {return (*x_values_)[i];}

    size_t size() const {return samples_.size();}
    const XY& get(size_t i) const {return *samples_[i];}

    void load(const XYSet& set)
    {
        spec_ = &set.spec();
        x_values_ = &set.x_values();
        samples_.clear();
        for (size_t i=0, s=set.size(); i<s; i++)
            add(set.get(i));
    }

    void add(const XY& xy)
    {
        samples_.push_back(&xy);
    }

    void clear()
    {
        spec_ = 0;
        x_values_ = 0;
        samples_.clear();
    }
};

// load liblinear format training samples
int load_liblinear(const char * filename, XYSet * set);
// load our format training samples
int load_gbdt(const char * filename, XYSet * set);
// load LECTOR 4.0 format training samples
// http://research.microsoft.com/en-us/um/beijing/projects/letor//letor4dataset.aspx
int load_lector4(const char * filename, XYSet * set, std::vector<size_t> * n_samples_per_query);

#endif// GBDT_TRAINING_SAMPLE_H
