#ifndef GBDT_JSON_H
#define GBDT_JSON_H

#include "sample.h"
#include <stdio.h>
#include <vector>

class TreeNodeBase;

int load_json(
    FILE * fp,
    double * y0,
    std::vector<TreeNodeBase *> * trees);

void save_json(
    FILE * fp,
    const XYSpec& spec,
    double y0,
    const std::vector<TreeNodeBase *>& trees);

#endif// GBDT_JSON_H
