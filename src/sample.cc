#include "sample.h"
#include "x.h"
#include <assert.h>
#include <string.h>
#include <algorithm>

static void skip_space(const char *& cur)
{
    if (*cur == ' ' || *cur == '\t')
        cur++;
}

// get some unique x values used when tree is being split
static void get_unique_x_values(
    XYSet * set,
    CompoundValueVector * x_values,
    size_t x_index,
    kXType x_type)
{
    x_values->clear();

    if (x_type == kXType_Numerical)
    {
        static const size_t MAX_UNIQUE_X_NUMERICAL = 100000;

        for (size_t i=0, s=std::min(MAX_UNIQUE_X_NUMERICAL, set->size()); i<s; i++)
            x_values->push_back(set->get(i).x(x_index));

        std::sort(x_values->begin(), x_values->end(), CompoundValueDoubleLess());

        double _min = x_values->front().d();
        // ensure the two continuous x values are not too near
        double delta = (1e-3);

        CompoundValueVector new_x_values;

        double last = _min - delta * 2;
        for (size_t i=0, s=x_values->size(); i<s; i++)
        {
            const CompoundValue& _now = (*x_values)[i];
            double _now_d = _now.d();
            if (_now_d - last > delta)
            {
                new_x_values.push_back(_now);
                last = _now_d;
            }
        }
        x_values->swap(new_x_values);
    }
    else
    {
        static const size_t MAX_UNIQUE_X_CATEGORY = 1024;

        for (size_t i=0, s=std::min(MAX_UNIQUE_X_CATEGORY, set->size()); i<s; i++)
            x_values->push_back(set->get(i).x(x_index));

        std::sort(x_values->begin(), x_values->end(), CompoundValueIntLess());
        x_values->erase(std::unique(x_values->begin(), x_values->end(), CompoundValueIntEqual()),
            x_values->end());
    }
}

static void get_unique_x_values(XYSet * set)
{
    set->x_values().resize(set->get_x_type_size());
    for (size_t i=0, s=set->spec().get_x_type_size(); i<s; i++)
        get_unique_x_values(set, &set->get_x_values(i), i, set->get_x_type(i));
}

class LibLinearLoader
{
private:
    size_t x_column_;
    size_t x_column_max_;

private:
    //+1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1
    //-1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1
    //+1 1:0.166667 2:1 3:-0.333333 4:-0.433962 5:-0.383562 6:-1 7:-1 8:0.0687023 9:-1 10:-0.903226 11:-1 12:-1 13:1
    int load_line(const char * line, XY * xy)
    {
        const char * cur = line;
        char * end;
        long x_index;
        double x_value;
        CompoundValue x;

        // weight is always 1.0 for liblinear
        xy->set_weight(1.0);

        // y
        if (strncmp(cur, "+1", 2) == 0)
            xy->y() = 1.0;
        else if (strncmp(cur, "-1", 2) == 0)
            xy->y() = -1.0;
        else
        {
            fprintf(stderr, "invalid y label\n");
            return -1;
        }
        cur += 2;
        skip_space(cur);

        // X
        for (;;)
        {
            if (*cur == 0 || *cur == '\n')
                break;

            x_index = strtol(cur, &end, 10);
            if (errno == ERANGE || cur == end)
            {
                fprintf(stderr, "invalid x index\n");
                return -1;
            }
            x_index--;

            if (*end != ':')
            {
                fprintf(stderr, "invalid separator: %c\n", *end);
                return -1;
            }
            cur = end + 1;

            x_value = strtod(cur, &end);
            if (errno == ERANGE || cur == end)
            {
                fprintf(stderr, "invalid x value\n");
                return -1;
            }
            cur = end;
            skip_space(cur);

            x.d() = x_value;
            if (xy->get_x_size() < (size_t)x_index + 1) 
                xy->resize_x((size_t)x_index + 1);
            xy->x(x_index) = x;
        }
        x_column_ = xy->get_x_size();
        if (x_column_max_ < x_column_)
            x_column_max_ = x_column_;
        return 0;
    }

public:
    LibLinearLoader() : x_column_(0), x_column_max_(0) {}

    int load(const char * filename, XYSet * set)
    {
        assert(filename);
        assert(set);

        FILE * fp = yfopen(filename, "r");
        if (fp == 0)
            return -1;
        ScopedFile fp_guard(fp);

        int length = 4096;
        char * line = (char *)xmalloc(length);
        ScopedPtrMalloc<char *> line_guard(line);
        char * to_read = line;
        int total_lines = 0, bad_lines = 0;

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

                XY xy;
                if (load_line(line, &xy) == -1)
                {
                    fprintf(stderr, "parse line failed:\n\"%s\"\n", line);
                    bad_lines++;
                }
                total_lines++;
                set->add(xy);
            }
        }

        if (x_column_max_ == 0)
        {
            printf("deduce spec failed\n");
            return 1;
        }

        printf("deduce spec: %d columns\n", (int)x_column_max_);
        printf("loaded %d training samples\n", (int)set->size());

        for (size_t i=0; i<x_column_max_; i++)
            set->add_x_type(kXType_Numerical);
        for (size_t i=0, s=set->size(); i<s; i++)
            set->get(i).resize_x(x_column_max_);

        if (set->size() == 0)
            return -1;

        get_unique_x_values(set);
        return 0;
    }
};

int load_liblinear(const char * filename, XYSet * set)
{
    LibLinearLoader loader;
    set->clear();
    return loader.load(filename, set);
}

class GBDTLoader
{
private:
    XYSpec spec_;

private:
    //#n c n n n n n n n n
    int load_spec(const char * line, XYSpec * spec)
    {
        const char * cur = line;
        if (*cur != '#')
        {
            fprintf(stderr, "invalid spec beginner\n");
            return -1;
        }
        cur++;
        skip_space(cur);

        for (;;)
        {
            char c = *cur;
            if (c == 0 || c == '\n')
                break;

            switch (c)
            {
            case 'n':
            case 'N':
                spec->add_x_type(kXType_Numerical);
                break;
            case 'c':
            case 'C':
                spec->add_x_type(kXType_Category);
                break;
            default:
                fprintf(stderr, "invalid spec description\n");
                return -1;
            }

            cur++;
            skip_space(cur);
        }
        return 0;
    }

    //0 61 0 60 468 36 0 52 1 1 0
    //0 57 1 233 145 5 0 107 20 2 0
    //1 w:5 53 0 313 6 0 0 4 0 2 0
    //1 w:4 33 0 1793 341 18 0 181 0 0 0
    //1 w:5 32 0 1784 366 15 0 166 0 0 0
    int load_xy(const char * line, XY * xy)
    {
        const char * cur = line;
        char * end;
        CompoundValue x;

        // y
        xy->y() = strtod(cur, &end);
        if (errno == ERANGE || cur == end)
        {
            fprintf(stderr, "invalid y value\n");
            return -1;
        }
        cur = end;
        skip_space(cur);

        // weight
        if (strncmp(cur, "w:", 2) == 0)
        {
            cur += 2;
            xy->set_weight(strtod(cur, &end));
            if (errno == ERANGE || cur == end)
            {
                fprintf(stderr, "invalid weight\n");
                return -1;
            }
            cur = end;
            skip_space(cur);
        }
        else
        {
            xy->set_weight(1.0);
        }

        // X
        xy->resize_x(spec_.get_x_type_size());
        for (size_t i=0, s=spec_.get_x_type_size(); i<s; i++)
        {
            kXType xtype = spec_.get_x_type(i);
            if (xtype == kXType_Numerical)
            {
                double value = strtod(cur, &end);
                if (errno == ERANGE || cur == end)
                {
                    fprintf(stderr, "invalid x value\n");
                    return -1;
                }
                xy->x(i).d() = value;
            }
            else
            {
                long value = strtol(cur, &end, 10);
                if (errno == ERANGE || cur == end)
                {
                    fprintf(stderr, "invalid x value\n");
                    return -1;
                }
                xy->x(i).i() = (int)value;
            }

            cur = end;
            skip_space(cur);
        }
        return 0;
    }

public:
    GBDTLoader() : spec_() {}

    int load(const char * filename, XYSet * set)
    {
        assert(filename);
        assert(set);

        FILE * fp = yfopen(filename, "r");
        if (fp == 0)
        {
            return -1;
        }
        ScopedFile fp_guard(fp);

        int length = 4096;
        char * line = (char *)xmalloc(length);
        ScopedPtrMalloc<char *> line_guard(line);
        char * to_read = line;
        int total_lines = 0, bad_lines = 0;
        int loaded_spec = 0;

        for (;;)
        {
            line[length-2] = 0;
            if (fgets(to_read, length-(to_read-line), fp) == 0)
            {
                break;
            }

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

                if (loaded_spec == 0)
                {
                    if (load_spec(line, &spec_) == -1)
                    {
                        fprintf(stderr, "load spec failed:\n\"%s\"\n", line);
                        return -1;
                    }
                    loaded_spec = 1;
                }
                else
                {
                    XY xy;
                    if (load_xy(line, &xy) == -1)
                    {
                        fprintf(stderr, "parse line failed:\n\"%s\"\n", line);
                        bad_lines++;
                    }
                    total_lines++;
                    set->add(xy);
                }
            }
        }

        printf("loaded spec: %d colunms\n", (int)spec_.get_x_type_size());
        printf("loaded %d training samples\n", (int)set->size());

        set->spec() = spec_;

        if (set->size() == 0)
            return -1;

        get_unique_x_values(set);
        return 0;
    }
};

int load_gbdt(const char * filename, XYSet * set)
{
    GBDTLoader loader;
    set->clear();
    return loader.load(filename, set);
}

class Lector4Loader
{
private:
    size_t x_column_;
    size_t x_column_max_;

private:
    //2 qid:10032 1:0.056537 2:0.000000 3:0.666667 4:1.000000 5:0.067138 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.058781 12:0.000000 13:0.591833 14:1.000000 15:0.066747 16:0.003980 17:0.000000 18:0.296296 19:0.200000 20:0.004012 21:0.946170 22:0.732324 23:0.520967 24:0.562389 25:0.000000 26:0.000000 27:0.000000 28:0.000000 29:0.504600 30:0.616488 31:0.215857 32:0.723049 33:1.000000 34:0.000000 35:0.000000 36:0.000000 37:0.953885 38:0.910033 39:0.490034 40:0.843384 41:0.000000 42:0.125000 43:0.000000 44:0.000000 45:0.000000 46:0.076923 #docid = GX029-35-5894638 inc = 0.0119881192468859 prob = 0.139842
    //0 qid:10032 1:0.279152 2:0.000000 3:0.000000 4:0.000000 5:0.279152 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.287177 12:0.000000 13:0.000000 14:0.000000 15:0.287226 16:0.014966 17:0.076923 18:0.333333 19:0.400000 20:0.015094 21:1.000000 22:0.834615 23:1.000000 24:0.623339 25:0.000000 26:0.000000 27:0.000000 28:0.000000 29:0.000000 30:0.000000 31:0.000000 32:0.000000 33:0.000000 34:0.000000 35:0.000000 36:0.000000 37:1.000000 38:1.000000 39:1.000000 40:0.906864 41:0.500000 42:0.000000 43:0.000000 44:0.002186 45:0.250000 46:1.000000 #docid = GX030-77-6315042 inc = 1 prob = 0.341364
    //0 qid:10035 1:0.891089 2:1.000000 3:1.000000 4:0.000000 5:1.000000 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.144213 12:1.000000 13:1.000000 14:0.000000 15:0.209717 16:0.654768 17:1.000000 18:1.000000 19:0.250000 20:0.680412 21:0.582831 22:0.569242 23:0.672193 24:0.724085 25:0.974209 26:1.000000 27:1.000000 28:1.000000 29:0.235213 30:0.000000 31:0.000000 32:0.000000 33:0.000000 34:0.000000 35:0.000000 36:0.000000 37:0.621058 38:0.610152 39:0.704347 40:0.743867 41:1.000000 42:0.207547 43:0.000000 44:0.008927 45:0.200000 46:0.166667 #docid = GX046-28-2590531 inc = 0.0121050330659901 prob = 0.119188
    //0 qid:10035 1:0.000000 2:0.000000 3:0.428571 4:0.000000 5:0.000000 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.183841 12:0.000000 13:0.779200 14:0.000000 15:0.237050 16:0.000000 17:0.166667 18:0.113636 19:0.416667 20:0.000000 21:0.847849 22:1.000000 23:0.344452 24:0.887347 25:0.000000 26:0.000000 27:0.000000 28:0.000000 29:1.000000 30:1.000000 31:1.000000 32:1.000000 33:0.000000 34:0.000000 35:0.000000 36:0.000000 37:0.900893 38:0.951122 39:0.437382 40:0.791401 41:1.000000 42:0.452830 43:0.000000 44:0.635237 45:1.000000 46:0.000000 #docid = GX058-84-15460908 inc = 1 prob = 0.115017
    int load_line(const char * line, XY * xy, long * qid)
    {
        const char * cur = line;
        char * end;
        long x_index;
        double x_value;
        CompoundValue x;

        // weight is always 1.0 for LECTOR 4.0
        xy->set_weight(1.0);

        // y
        // Labels in LECTOR 4.0 are integers.
        long _label = strtol(cur, &end, 10);
        if (errno == ERANGE || cur == end || _label < 0)
        {
            fprintf(stderr, "invalid y label\n");
            return -1;
        }
        xy->label() = (size_t)_label;
        cur = end + 1;
        skip_space(cur);

        // qid
        if (strncmp(cur, "qid:", 4) != 0)
        {
            fprintf(stderr, "no qid\n");
            return -1;
        }
        cur += 4;
        *qid = strtol(cur, &end, 10);
        if (errno == ERANGE || cur == end)
        {
            fprintf(stderr, "invalid qid\n");
            return -1;
        }
        cur = end + 1;
        skip_space(cur);

        // X
        for (;;)
        {
            if (*cur == 0 || *cur == '\n' || *cur == '#')
                break;

            x_index = strtol(cur, &end, 10);
            if (errno == ERANGE || cur == end)
            {
                fprintf(stderr, "invalid x index\n");
                return -1;
            }
            x_index--;

            if (*end != ':')
            {
                fprintf(stderr, "invalid separator: %c\n", *end);
                return -1;
            }
            cur = end + 1;

            x_value = strtod(cur, &end);
            if (errno == ERANGE || cur == end)
            {
                fprintf(stderr, "invalid x value\n");
                return -1;
            }
            cur = end;
            skip_space(cur);

            x.d() = x_value;
            if (xy->get_x_size() < (size_t)x_index + 1) 
                xy->resize_x((size_t)x_index + 1);
            xy->x(x_index) = x;
        }
        x_column_ = xy->get_x_size();
        if (x_column_max_ < x_column_)
            x_column_max_ = x_column_;
        return 0;
    }

public:
    Lector4Loader() : x_column_(0), x_column_max_(0) {}

    int load(const char * filename, XYSet * set, std::vector<size_t> * n_samples_per_query)
    {
        assert(filename);
        assert(set);

        FILE * fp = yfopen(filename, "r");
        if (fp == 0)
            return -1;
        ScopedFile fp_guard(fp);

        int length = 4096;
        char * line = (char *)xmalloc(length);
        ScopedPtrMalloc<char *> line_guard(line);
        char * to_read = line;
        int total_lines = 0, bad_lines = 0;

        bool first_qid = true;
        long qid = -1;
        size_t qid_count = 0;
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

                XY xy;
                long previous_qid = qid;
                if (load_line(line, &xy, &qid) == -1)
                {
                    fprintf(stderr, "parse line failed:\n\"%s\"\n", line);
                    bad_lines++;
                }

                if (first_qid)
                {
                    first_qid = false;
                    qid_count = 1;
                }
                else
                {
                    if (qid == previous_qid)
                    {
                        qid_count++;
                    }
                    else
                    {
                        n_samples_per_query->push_back(qid_count);
                        qid_count = 1;
                    }
                }

                total_lines++;
                set->add(xy);
            }
        }
        n_samples_per_query->push_back(qid_count);

        if (x_column_max_ == 0)
        {
            printf("deduce spec failed\n");
            return 1;
        }

        printf("deduce spec: %d columns\n", (int)x_column_max_);
        printf("loaded %d training samples, %d queries\n",
            (int)set->size(),
            (int)n_samples_per_query->size());

        for (size_t i=0; i<x_column_max_; i++)
            set->add_x_type(kXType_Numerical);
        for (size_t i=0, s=set->size(); i<s; i++)
            set->get(i).resize_x(x_column_max_);

        if (set->size() == 0)
            return -1;

        get_unique_x_values(set);
        return 0;
    }
};

int load_lector4(const char * filename, XYSet * set, std::vector<size_t> * n_samples_per_query)
{
    Lector4Loader loader;
    set->clear();
    n_samples_per_query->clear();
    return loader.load(filename, set, n_samples_per_query);
}
