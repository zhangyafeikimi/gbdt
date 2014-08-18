#include "lm.h"
#include "lm-scorer.h"
#include "sample.h"
#include "x.h"
#include <stdio.h>
#include <functional>

int main()
{
    XYSet set;
    std::vector<size_t> n_samples_per_query;
    // lector4-train.txt is large, so it is not in git.
    // It can be extracted from the following URLs:
    // http://research.microsoft.com/en-us/um/beijing/projects/letor//LETOR4.0/Data/MQ2007.rar
    // http://research.microsoft.com/en-us/um/beijing/projects/letor//LETOR4.0/Data/MQ2008.rar
#if defined _MSC_VER
    load_lector4("../data/lector4-train.txt", &set, &n_samples_per_query);
#else
    load_lector4("./data/lector4-train.txt", &set, &n_samples_per_query);
#endif

    TreeParam param;
    param.verbose = 1;
    param.max_level = 5;
    param.max_leaf_number = 20;
    param.min_values_in_leaf = 10;
    param.tree_number = 100;
    param.learning_rate = 0.01;

    param.gbdt_sample_rate = 1.0;
    param.lm_metric = "ndcg";
    param.lm_ndcg_k = 8;

    LambdaMARTTrainer trainer(set, n_samples_per_query, param);
    trainer.train();

    FILE * output = xfopen("output.lambdamart.json", "w");
    trainer.save_json(output);
    fclose(output);

    LambdaMARTPredictor predictor;
    FILE * input = xfopen("output.lambdamart.json", "r");
    predictor.load_json(input);
    fclose(input);

    size_t ndcg_k = param.lm_ndcg_k;
    const NDCGScorer ndcg_scorer(ndcg_k);
    std::vector<size_t> labels;// manual labels
    std::vector<size_t> new_labels;// manual labels sorted by 'scores'
    std::vector<double> scores;// model scores
    std::vector<size_t> indices;
    double ndcg;
    double dcg;
    double idcg;
    char dcg_name[16];
    sprintf(dcg_name, "dcg@%d", (int)ndcg_k);

    size_t begin = 0;
    for (size_t i=0, s=n_samples_per_query.size(); i<s; i++)
    {
        // for each query-result list
        const XY * results = &set.sample()[begin];
        size_t result_size = n_samples_per_query[i];

        labels.clear(); labels.reserve(result_size);
        scores.clear(); scores.reserve(result_size);
        for (size_t j=0; j<result_size; j++)
        {
            const XY& xy = results[j];
            labels.push_back((size_t)xy.y());
            scores.push_back(trainer.predict(xy.X()));
        }

        sort_indices(&scores[0], scores.size(), &indices, std::greater<double>());
        new_labels.clear(); new_labels.reserve(result_size);

        for (size_t j=0; j<result_size; j++)
            new_labels.push_back(labels[indices[j]]);

        ndcg_scorer.get_score(new_labels, &ndcg, &dcg, &idcg);
        printf("n%s=%lf, %s=%lf, i%s=%lf\n", dcg_name, ndcg, dcg_name, dcg, dcg_name, idcg);

        begin += result_size;
    }

    return 0;
}
