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
        "        see README.md for specifications\n"
        "        see data/heart_scale.conf or data/weibo.conf for example\n"
        );
}

struct TreeParamSpec
{
    const char * type_name;
    const char * name;
    void * v;
    void (* assign)(const std::string& s, void * v);
    void (* check)(void * v);
    bool _set;
};

#define DECLARE_PARAM(param, type_name, name) \
{#type_name, #name, (void *)(&param->name), assign_##type_name, 0, false}
#define DECLARE_PARAM2(param, type_name, name) \
{#type_name, #name, (void *)(&param->name), assign_##type_name, check_##name, false}

static void assign_int(const std::string& s, void * v)
{
    *(int *)v = xatoi(s.c_str());
}

static void assign_size_t(const std::string& s, void * v)
{
    *(size_t *)v = (size_t)xatoi(s.c_str());
}

static void assign_double(const std::string& s, void * v)
{
    *(double *)v = xatof(s.c_str());
}

static void assign_std_string(const std::string& s, void * v)
{
    *(std::string *)v = s;
}

static void check_min_values_in_leaf(void * v)
{
    if (*(size_t *)v < 1)
    {
        fprintf(stderr, "invalid \"min_values_in_leaf\", it should be >= 1\n");
        exit(1);
    }
}

static void check_loss(void * v)
{
    std::string loss = *(std::string *)v;
    if (loss != "ls" && loss != "lad" && loss != "logistic")
    {
        fprintf(stderr, "invalid \"loss\", it should be \"ls\", \"lad\" or \"logistic\"\n");
        exit(1);
    }
}

static void check_training_sample_format(void * v)
{
    std::string format = *(std::string *)v;
    if (format != "liblinear" && format != "gbdt")
    {
        fprintf(stderr, "invalid \"training_sample_format\", it should be \"liblinear\" or \"gbdt\"\n");
        exit(1);
    }
}

class TreeParamLoader
{
private:
    int add_key_value(
        TreeParam * param,
        TreeParamSpec * specs,
        size_t spec_length,
        const std::string& key,
        const std::string& value)
    {
        for (size_t i=0; i<spec_length; i++)
        {
            TreeParamSpec& spec = specs[i];
            if (key == spec.name)
            {
                spec.assign(value, spec.v);
                if (spec.check)
                    spec.check(spec.v);
                spec._set = true;
                return 0;
            }
        }

        return -1;
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

        TreeParamSpec specs[] =
        {
            DECLARE_PARAM(param, int, verbose),
            DECLARE_PARAM(param, size_t, max_level),
            DECLARE_PARAM(param, size_t, max_leaf_number),
            DECLARE_PARAM2(param, size_t, min_values_in_leaf),
            DECLARE_PARAM(param, size_t, tree_number),
            DECLARE_PARAM(param, double, learning_rate),
            DECLARE_PARAM(param, double, sample_rate),
            DECLARE_PARAM2(param, std_string, loss),
            DECLARE_PARAM(param, std_string, training_sample),
            DECLARE_PARAM2(param, std_string, training_sample_format),
            DECLARE_PARAM(param, std_string, model),
        };
        size_t spec_length = sizeof(specs)/sizeof(specs[0]);

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

                if (add_key_value(param, specs, spec_length, key, value) == -1)
                {
                    fprintf(stderr, "invalid key or value:\"%s\", \"%s\"\n", key.c_str(), value.c_str());
                    return -1;
                }
            }
        }

        for (size_t i=0; i<spec_length; i++)
        {
            const TreeParamSpec& spec = specs[i];
            if (!spec._set)
            {
                fprintf(stderr, "\"%s\" is not set in \"%s\"\n", spec.name, filename);
                return -1;
            }
        }

        return 0;
    }
};

int parse_tree_param(int argc, char ** argv, TreeParam * param)
{
    assert(param);

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
