#include "x.h"
#include "gbdt.h"

int main(int argc, char ** argv)
{
    TreeParam param;
    if (gbdt_parse_tree_param(argc, argv, &param) == -1)
        return 1;

    XYSet set;
    if (param.training_sample_format == "liblinear")
    {
        if (load_liblinear(param.training_sample.c_str(), &set) == -1)
            return 2;
    }
    else
    {
        if (load_gbdt(param.training_sample.c_str(), &set) == -1)
            return 2;
    }

    GBDTPredictor predictor;
    FILE * input = xfopen(param.model.c_str(), "r");
    predictor.load_json(input);
    fclose(input);

    for (size_t i=0, s=set.size(); i<s; i++)
    {
        const XY& xy = set.get(i);
        const CompoundValueVector& X = xy.X();
        double y = xy.y();
        printf("%lf should be near to %lf\n", predictor.predict(X), y);
    }

    return 0;
}
