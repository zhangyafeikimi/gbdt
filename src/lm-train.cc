#include "lm.h"
#include "sample.h"
#include "x.h"
#include <stdio.h>

int main()
{
    XYSet set;
    std::vector<size_t> n_samples_per_query;
    load_lector4("test.txt", &set, &n_samples_per_query);

    TreeParam param;
    param.verbose = 1;
    param.max_level = 5;
    param.max_leaf_number = 20;
    param.min_values_in_leaf = 10;
    param.tree_number = 400;
    param.learning_rate = 0.01;

    param.gbdt_sample_rate = 1.0;
    param.lm_metric = "ndcg";
    param.lm_ndcg_k = 5;

    {
        LambdaMARTTrainer trainer(set, n_samples_per_query, param);
        trainer.train();

        FILE * output = xfopen("output.lambdamart.json", "w");
        trainer.save_json(output);
        fclose(output);

        LambdaMARTPredictor predictor;
        FILE * input = xfopen("output.lambdamart.json", "r");
        predictor.load_json(input);
        fclose(input);

        for (size_t i=0, s=set.size(); i<s; i++)
        {
            const XY& xy = set.get(i);
            const CompoundValueVector& X = xy.X();
            double y = xy.y();
            printf("score=%lf=%lf, label=%d\n",
                trainer.predict(X), predictor.predict(X), (int)y);
        }
    }

    return 0;
}
