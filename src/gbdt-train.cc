#include "x.h"
#include "gbdt.h"

int main(int argc, char ** argv)
{
    TreeParam param;
    if (parse_tree_param(argc, argv, &param) == -1)
        return 1;

    XYSet set;
    if (load_liblinear(param.training_sample.c_str(), &set) == -1)
        return 2;

    GBDTTrainer trainer(set, param);
    trainer.train();

    FILE * output = xfopen(param.model.c_str(), "w");
    trainer.save_json(output);
    fclose(output);

    return 0;
}
