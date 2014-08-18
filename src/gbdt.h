#ifndef GBDT_GBDT_H
#define GBDT_GBDT_H

#include "param.h"
#include "sample.h"
#include <stdio.h>
#include <vector>

class TreeNodeBase;

class GBDTPredictor
{
protected:
    double y0_;
    std::vector<TreeNodeBase *> trees_;
public:
    GBDTPredictor() {}
    virtual ~GBDTPredictor() {clear();}
    double predict(const CompoundValueVector& X) const;
    double predict_logistic(const CompoundValueVector& X) const;
    int load_json(FILE * fp);
    void clear();
};

class GBDTTrainer : public GBDTPredictor
{
private:
    const XYSet& full_set_;
    const TreeParam& param_;
    std::vector<double> full_fx_;
    const TreeNodeBase * holder_;
    double total_loss() const;
    void dump_feature_importance() const;
public:
    GBDTTrainer(const XYSet& set, const TreeParam& param);
    virtual ~GBDTTrainer();
    void train();
    void save_json(FILE * fp) const;
};

#endif// GBDT_GBDT_H
