#ifndef GBDT_GAIN_H
#define GBDT_GAIN_H

#include "param.h"
#include "sample.h"
#include <stdio.h>

class TreeGainNode;
class TreeGain
{
private:
    const XYSet& set_;
    const TreeParam& param_;
    TreeGainNode * root_;

public:
    TreeGain(const XYSet& set, const TreeParam& param);
    ~TreeGain();
    void train();
    double predict(const CompoundValueVector& X) const;
    void dump_dot(FILE * fp) const;
};

#endif// GBDT_GAIN_H
