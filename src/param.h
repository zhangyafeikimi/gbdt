#ifndef GBDT_PARAM_H
#define GBDT_PARAM_H

#include <stddef.h>

struct TreeParam
{
    int verbose;

    size_t max_level;
    size_t max_leaf_number;
    size_t max_x_values_number;
    double leaf_threshold;

    size_t gbdt_tree_number;
    double gbdt_learning_rate;
    double gbdt_sample_rate;

    TreeParam() {}
};

#endif// GBDT_PARAM_H
