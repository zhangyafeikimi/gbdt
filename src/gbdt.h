#ifndef GBDT_GBDT_H
#define GBDT_GBDT_H

#include "param.h"
#include "sample.h"
#include <stdio.h>
#include <vector>

class TreeLossNode;

class GBDTPredictor
{
private:
    static const TreeParam empty_param_;
protected:
    double y0_;
    std::vector<TreeLossNode *> trees_;
public:
    GBDTPredictor();
    virtual ~GBDTPredictor();
    double predict(const CompoundValueVector& X) const;
    int load_json(FILE * fp);
    void clear();
};

class GBDTTrainer : public GBDTPredictor
{
private:
    const XYSet& full_set_;
    const TreeParam& param_;
    std::vector<double> full_residual_;
    double ls_loss() const;
    double lad_loss() const;
    double total_loss() const;
    void dump_feature_importance() const;
public:
    GBDTTrainer(const XYSet& set, const TreeParam& param);
    virtual ~GBDTTrainer();
    void train();
    void save_json(FILE * fp) const;
};

#endif// GBDT_GBDT_H
