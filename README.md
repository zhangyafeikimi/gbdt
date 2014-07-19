ml-gbdt
=====================

**ml-gbdt** is a GBDT(MART) training and predicting package.

Compiling
---------
**make** it or compile it with **Visual Studio**.

Training
--------
>./gbdt-train -c [configuration file]

Predicting
--------
>./gbdt-predict -c [configuration file]

Configuration File
------------------
###An Example

>verbose = 1

>max_level = 5

>max_leaf_number = 20

>max_x_values_number = 200

>leaf_threshold = 0.75

>gbdt_tree_number = 400

>gbdt_learning_rate = 0.1

>gbdt_sample_rate = 0.9

>gbdt_loss = ls

>training_sample = input

>training_sample_format = liblinear

>model = output.json

###Specification

All fields and values are case-sensitive.

####verbose
0, print least information
1, print extra information

####max_level
Max level of all decision trees.

####max_leaf_number
Max number of leaf node in all decision trees.

####max_x_values_number
Max unique values of one feature used when a tree node is being split by this feature.

####leaf_threshold
It should be in [0.0, 1.0].

It is not used in gbdt, used only in decision tree.

####gbdt_tree_number
Number of trees.

####gbdt_learning_rate
Learning rate, should be in [0.0, 1.0], defined at **Friedman (March 1999)**.

####gbdt_sample_rate
Sample rate, should be in [0.0, 1.0], defined at **Friedman (March 1999)**.

####gbdt_loss
GBDT loss type, can be "ls" or "lad".

LS and LAD loss are defined at **Friedman (February 1999)**

> **NOTE:**
>
> LAD loss maybe numerical unstable and cause invalid feature importance. LS loss is numerically smoother.
>
> Training a GBDT model with LAD loss is slower then the one with LS loss.
>
> LAD loss is only suitable for binary classification.
>
> **All in all, LS loss is highly encouraged.**

####training_sample
Filename of training samples.

####training_sample_format
Format of training sample, can be "liblinear" or "gbdt".

**ml-gbdt** is fully compatible with [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) format. An example is:

>+1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1

>-1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1

>+1 1:0.166667 2:1 3:-0.333333 4:-0.433962 5:-0.383562 6:-1 7:-1 8:0.0687023 9:-1 10:-0.903226 11:-1 12:-1 13:1

While I have defined another format for some reasons below.

An example of gbdt format is:

> \#n c n n n n n n n n

> 0 61 0 60 468 36 0 52 1 1 0

> 0 57 1 233 145 5 0 107 20 2 0

> 1 w:5.5 53 0 313 6 0 0 4 0 2 0

> 1 w:4 33 0 1793 341 18 0 181 0 0 0



> **Some Explanations:**

> The first line shows there are 10 features, the 2nd of which is a category feature, others are numerical features.

> Values of category features must be integers.

> Values of numerical features and "y" can be double floats or integers(treated as double float internally).

> "y" can be 0/1 to model a binary classification problem(**NOTE**: liblinear is -1/1), or any real numbers for regression.

> From the 2nd line on, the 1st column is the "y" values, others are ordered "x" values.

> The 4th and 5th line contains "w:5.5", "w:4" respectively. 5.5 and 4 are weights of the two training samples.
> Default weights are 1.0.

> **Advantages:**

> Category features.

> Feature weights.

> Classification and regression.


####model
Filename of the model, the output for "gbdt-train" and the input for "gbdt-predict".
It is in json and very easy to understand.

Others
-----
### Classical Decision Tree
class "TreeGain" in gain.h is an implementation of classical decision tree based on information gain.
It can be accessed only through libgbdt. No command line tools supporting it.

### json2cxx.py
"json2cxx.py" lies in directory "tools".
It can be used to convert a model(json) to a c++ predicting function, so that an interpreter for predicting is avoided.

Reference
---------
[Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine." (February 1999)](http://www-stat.stanford.edu/~jhf/ftp/trebst.pdf)

[Friedman, J. H. "Stochastic Gradient Boosting." (March 1999)](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)
