#include "param.h"
#include "x.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void print_usage(const char * program, FILE * fp)
{
    fprintf(fp, "%s, A GBDT(MART) training and predicting package.\n"
        "Copyright (C) 2012-2014 Zhang Yafei, All rights reserved.\n",
        program);
    fprintf(fp,
        "usage:\n"
        "    -h, show help\n"
        "    -c [configuration file], specify the configuration file\n"
        "        configuration file example:\n"
        "        -------------------------\n"
        "        verbose = 1\n"
        "        max_level = 5\n"
        "        max_leaf_number = 20\n"
        "        max_x_values_number = 200\n"
        "        leaf_threshold = 0.75\n"
        "        gbdt_tree_number = 400\n"
        "        gbdt_learning_rate = 0.1\n"
        "        gbdt_sample_rate = 0.9\n"
        "        gbdt_loss = ls\n"
        "        training_sample = input\n"
        "        training_sample_format = liblinear\n"
        "        model = output.json\n"
        "        -------------------------\n");
}

class TreeParamLoader
{
private:
    int add_key_value(TreeParam * param, const std::string& key, const std::string& value)
    {
        if (key == "verbose")
            param->verbose = xatoi(value.c_str());
        else if (key == "max_level")
            param->max_level = xatoi(value.c_str());
        else if (key == "max_leaf_number")
            param->max_leaf_number = xatoi(value.c_str());
        else if (key == "max_x_values_number")
            param->max_x_values_number = xatoi(value.c_str());
        else if (key == "leaf_threshold")
            param->leaf_threshold = xatof(value.c_str());
        else if (key == "gbdt_tree_number")
            param->gbdt_tree_number = xatoi(value.c_str());
        else if (key == "gbdt_learning_rate")
            param->gbdt_learning_rate = xatof(value.c_str());
        else if (key == "gbdt_sample_rate")
            param->gbdt_sample_rate = xatof(value.c_str());
        else if (key == "gbdt_loss")
        {
            if (value != "ls" && value != "lad")
            {
                fprintf(stderr, "invalid \"gbdt_loss\", it should be \"ls\" or \"lad\"\n");
                return -1;
            }
            param->gbdt_loss = value;
        }
        else if (key == "training_sample")
            param->training_sample = value;
        else if (key == "training_sample_format")
        {
            if (value != "liblinear" && value != "gbdt")
            {
                fprintf(stderr, "invalid \"training_sample_format\", it should be \"liblinear\" or \"gbdt\"\n");
                return -1;
            }
            param->training_sample_format = value;
        }
        else if (key == "model")
            param->model = value;
        return 0;
    }

    void trim_assign(const char * src, std::string * dest)
    {
        dest->clear();
        for (;; src++)
        {
            if (*src == 0)
                return;

            if (*src != ' ' && *src != '\t')
                dest->push_back(*src);
        }
    }

    int load_line(const char * line, std::string * key, std::string * value)
    {
        std::string line_bak = line;
        line_bak.resize(line_bak.size() - 1);// pop \n
        size_t pos = line_bak.find_first_of('=');
        if (pos == std::string::npos)
            return -1;

        line_bak[pos] = 0;
        trim_assign(&line_bak[0], key);
        trim_assign(&line_bak[pos+1], value);

        if (key->empty() || value->empty())
            return -1;
        return 0;
    }

public:
    int load(const char * filename, TreeParam * param)
    {
        assert(filename);
        assert(param);

        FILE * fp = yfopen(filename, "r");
        if (fp == 0)
            return -1;
        ScopedFile fp_guard(fp);

        int length = 4096;
        char * line = (char *)xmalloc(length);
        ScopedPtrMalloc<char *> line_guard(line);
        char * to_read = line;

        for (;;)
        {
            line[length-2] = 0;
            if (fgets(to_read, length-(to_read-line), fp) == 0)
                break;

            if (line[length-2] != 0)
            {
                int line_length = length - 1;
                length = length * 2;
                line = (char *)xrealloc(line, length);
                line_guard.set_for_realloc(line);
                to_read = line + line_length;
            }
            else
            {
                to_read = line;

                std::string key, value;
                if (load_line(line, &key, &value) == -1)
                {
                    fprintf(stderr, "parse line failed:\n\"%s\"\n", line);
                    return -1;
                }

                if (add_key_value(param, key, value) == -1)
                {
                    fprintf(stderr, "invalid key or value:\"%s\", \"%s\"\n", key.c_str(), value.c_str());
                    return -1;
                }
            }
        }

        return 0;
    }
};

int parse_tree_param(int argc, char ** argv, TreeParam * param)
{
    assert(param);
    // set default values
    param->verbose = 1;
    param->max_level = 5;
    param->max_leaf_number = 20;
    param->max_x_values_number = 200;
    param->leaf_threshold = 0.75;
    param->gbdt_tree_number = 400;
    param->gbdt_learning_rate = 0.1;
    param->gbdt_sample_rate = 0.9;
    param->gbdt_loss = "ls";
    param->training_sample = "input";
    param->training_sample_format = "liblinear";
    param->model = "output.json";

    std::string config_filename;
    for (int i=1; i<argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0], stderr);
            return -1;
        }
        else if (strcmp(argv[i], "-c") == 0 && i+1<argc)
        {
            config_filename = argv[i+1];
            i++;
        }
        else
        {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return -1;
        }
    }

    if (config_filename.empty())
    {
        print_usage(argv[0], stderr);
        return -1;
    }

    TreeParamLoader loader;
    return loader.load(config_filename.c_str(), param);
}
