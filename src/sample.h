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

// training sample specification
class XYSpec
{
private:
    std::vector<kXType> xtypes_;

public:
    size_t get_xtype_size() const {return xtypes_.size();}
    kXType get_xtype(size_t i) const {return xtypes_[i];}
    void add_xtype(kXType xtype) {xtypes_.push_back(xtype);}
    void clear_xtype() {xtypes_.clear();}
};

class CompoundValue
{
private:
    union
    {
        double d;
        int i;
    } value_;

public:
    CompoundValue() {value_.d = 0.0;}
    double& d() {return value_.d;}
    double d() const {return value_.d;}
    int& i() {return value_.i;}
    int i() const {return value_.i;}
};

typedef std::vector<CompoundValue> CompoundValueVector;

class CompoundValueVectorBuilder
{
private:
    CompoundValueVector vector_;
public:
    CompoundValueVectorBuilder& d(double _d)
    {
        CompoundValue v;
        v.d() = _d;
        vector_.push_back(v);
        return *this;
    }

    CompoundValueVectorBuilder& i(int _i)
    {
        CompoundValue v;
        v.i() = _i;
        vector_.push_back(v);
        return *this;
    }

    CompoundValueVector build()
    {
        CompoundValueVector empty;
        empty.swap(vector_);
        return empty;
    }
};

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
    CompoundValueVector X_;
    double y_;
    double weight_;

public:
    size_t get_x_size() const {return X_.size();}
    CompoundValue& x(size_t i) {return X_[i];}
    const CompoundValue& x(size_t i) const {return X_[i];}
    const CompoundValueVector& X() const {return X_;}
    void add_x(const CompoundValue& _x) {X_.push_back(_x);}
    void resize_x(size_t s) {X_.resize(s);}

    double& y() {return y_;}
    double y() const {return y_;}

    double& weight() {return weight_;}
    double weight() const {return weight_;}
};

// a set of training samples
class XYSet
{
private:
    XYSpec spec_;
    std::vector<XY> samples_;

public:
    XYSpec& spec() {return spec_;}
    const XYSpec& spec() const {return spec_;}

    size_t get_xtype_size() const {return spec_.get_xtype_size();}
    kXType get_xtype(size_t i) const {return spec_.get_xtype(i);}
    void add_xtype(kXType xtype) {spec_.add_xtype(xtype);}

    size_t size() const {return samples_.size();}
    XY& get(size_t i) {return samples_[i];}
    const XY& get(size_t i) const {return samples_[i];}
    void add(const XY& xy) {samples_.push_back(xy);}

    void clear()
    {
        spec_.clear_xtype();
        samples_.clear();
    }
};

// external reference to a set of training samples
class XYSetRef
{
private:
    const XYSpec * spec_;
    std::vector<const XY *> samples_;

public:
    XYSetRef() {clear();}

    const XYSpec *& spec() {return spec_;}
    const XYSpec * spec() const {return spec_;}

    size_t get_xtype_size() const {return spec_->get_xtype_size();}
    kXType get_xtype(size_t i) const {return spec_->get_xtype(i);}

    size_t size() const {return samples_.size();}
    const XY& get(size_t i) const {return *samples_[i];}

    void load(const XYSet& set)
    {
        spec_ = &set.spec();
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
        samples_.clear();
    }
};

// load liblinear format training sample
int load_liblinear(const char * filename, XYSet * set);
// load our format training sample
int load_gbdt(const char * filename, XYSet * set);

#endif// GBDT_TRAINING_SAMPLE_H
