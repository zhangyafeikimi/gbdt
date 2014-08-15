#ifndef GBDT_PARAM_H
#define GBDT_PARAM_H

#include <stddef.h>
#include <string>

// see README.md for specifications
struct TreeParam
{
    int verbose;

    size_t max_level;
    size_t max_leaf_number;
    size_t min_values_in_leaf;

    size_t tree_number;
    double learning_rate;

    std::string training_sample;
    std::string training_sample_format;
    std::string model;

    double gbdt_sample_rate;
    std::string gbdt_loss;

    std::string lm_metric;
    size_t lm_ndcg_k;

    TreeParam() {}
};

int gbdt_parse_tree_param(int argc, char ** argv, TreeParam * param);
int lm_parse_tree_param(int argc, char ** argv, TreeParam * param);

#endif// GBDT_PARAM_H
