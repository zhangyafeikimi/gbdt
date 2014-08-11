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
    size_t max_x_values_number;
    size_t min_values_in_leaf;

    size_t tree_number;
    double learning_rate;
    double sample_rate;
    std::string loss;

    std::string training_sample;
    std::string training_sample_format;
    std::string model;

    TreeParam() {}
};

int parse_tree_param(int argc, char ** argv, TreeParam * param);

#endif// GBDT_PARAM_H
