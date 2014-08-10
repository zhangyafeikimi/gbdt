#include "x.h"
#include "gbdt.h"

int main()
{
    XYSet set;
#if defined _MSC_VER
    load_liblinear("../data/heart_scale.txt", &set);
#else
    load_liblinear("./data/heart_scale.txt", &set);
#endif

    TreeParam param;
    param.verbose = 1;
    param.max_level = 5;
    param.max_leaf_number = 20;
    param.max_x_values_number = 200;
    param.min_values_in_leaf = 10;
    param.gbdt_tree_number = 400;
    param.gbdt_learning_rate = 0.1;
    param.gbdt_sample_rate = 0.9;

    param.gbdt_loss = "logistic";
    {
        GBDTTrainer trainer(set, param);
        trainer.train();

        FILE * output = xfopen("output.logistic.json", "w");
        trainer.save_json(output);
        fclose(output);

        GBDTPredictor predictor;
        FILE * input = xfopen("output.logistic.json", "r");
        predictor.load_json(input);
        fclose(input);

        for (size_t i=0, s=set.size(); i<s; i++)
        {
            const XY& xy = set.get(i);
            const CompoundValueVector& X = xy.X();
            double y = xy.y();
            printf("p(y=1|x)=%lf=%lf, y=%lf\n",
                trainer.predict_logistic(X), predictor.predict_logistic(X), y);
        }
    }

    //param.gbdt_loss = "ls";
    //{
    //    GBDTTrainer trainer(set, param);
    //    trainer.train();

    //    FILE * output = xfopen("output.ls.json", "w");
    //    trainer.save_json(output);
    //    fclose(output);

    //    GBDTPredictor predictor;
    //    FILE * input = xfopen("output.ls.json", "r");
    //    predictor.load_json(input);
    //    fclose(input);

    //    for (size_t i=0, s=set.size(); i<s; i++)
    //    {
    //        const XY& xy = set.get(i);
    //        const CompoundValueVector& X = xy.X();
    //        double y = xy.y();
    //        printf("F(x)=%lf=%lf, y=%lf\n",
    //            trainer.predict(X), predictor.predict(X), y);
    //    }
    //}

    //param.gbdt_loss = "lad";
    //{
    //    GBDTTrainer trainer(set, param);
    //    trainer.train();

    //    FILE * output = xfopen("output.lad.json", "w");
    //    trainer.save_json(output);
    //    fclose(output);

    //    GBDTPredictor predictor;
    //    FILE * input = xfopen("output.lad.json", "r");
    //    predictor.load_json(input);
    //    fclose(input);

    //    for (size_t i=0, s=set.size(); i<s; i++)
    //    {
    //        const XY& xy = set.get(i);
    //        const CompoundValueVector& X = xy.X();
    //        double y = xy.y();
    //        printf("F(x)=%lf=%lf, y=%lf\n",
    //            trainer.predict(X), predictor.predict(X), y);
    //    }
    //}

    return 0;
}
