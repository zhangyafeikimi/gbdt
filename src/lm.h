#ifndef GBDT_LAMBDA_MART_H
#define GBDT_LAMBDA_MART_H

#include "param.h"
#include "sample.h"
#include <stdio.h>
#include <vector>

class TreeNodeBase;
class LambdaMARTNode;
class NDCGScorer;

class LambdaMARTPredictor
{
protected:
    double y0_;
    std::vector<TreeNodeBase *> trees_;
public:
    LambdaMARTPredictor() {}
    virtual ~LambdaMARTPredictor() {clear();}
    double predict(const CompoundValueVector& X) const;
    int load_json(FILE * fp);
    void clear();
};

class LambdaMARTTrainer : public LambdaMARTPredictor
{
private:
    const XYSet& full_set_;
    const TreeParam& param_;
    std::vector<double> full_fx_;
    const LambdaMARTNode * holder_;
    const NDCGScorer * scorer_;
public:
    LambdaMARTTrainer(
        const XYSet& set,
        const std::vector<size_t>& n_samples_per_query,
        const TreeParam& param);
    virtual ~LambdaMARTTrainer();
    void train();
    void save_json(FILE * fp) const;
};

#endif// GBDT_LAMBDA_MART_H
