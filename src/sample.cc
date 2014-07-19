#include "sample.h"
#include "x.h"
#include <assert.h>
#include <string.h>

class Loader
{
public:
    static void skip_space(const char *& cur)
    {
        if (*cur == ' ' || *cur == '\t')
            cur++;
    }

    virtual ~Loader() {}
    virtual int load(const char * filename, XYSet * set) = 0;
};

class LibLinearLoader : public Loader
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

        // y
        if (strncmp(cur, "+1", 2) == 0)
            xy->y() = 1.0;
        else if (strncmp(cur, "-1", 2) == 0)
            xy->y() = 0.0;
        else
        {
            fprintf(stderr, "invalid y label\n");
            return -1;
        }
        cur += 2;
        skip_space(cur);

        // weight
        xy->weight() = 1.0;

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

    virtual int load(const char * filename, XYSet * set)
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
                if (load_line(line, &xy) != 0)
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
        printf("loaded %d training samples, %d bad training samples\n", total_lines, bad_lines);

        for (size_t i=0; i<x_column_max_; i++)
            set->add_xtype(kXType_Numerical);
        for (size_t i=0, s=set->size(); i<s; i++)
            set->get(i).resize_x(x_column_max_);

        return 0;
    }
};

class GBDTLoader : public Loader
{
private:
    XYSpec spec_;

private:
    //#n c n n n n n n n n
    virtual int load_spec(const char * line, XYSpec * spec)
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
                spec->add_xtype(kXType_Numerical);
                break;
            case 'c':
            case 'C':
                spec->add_xtype(kXType_Category);
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
    int load_x(const char * line, XY * xy)
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
            xy->weight() = strtod(cur, &end);
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
            xy->weight() = 1.0;
        }

        // X
        xy->resize_x(spec_.get_xtype_size());
        for (size_t i=0, s=spec_.get_xtype_size(); i<s; i++)
        {
            kXType xtype = spec_.get_xtype(i);
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

    virtual int load(const char * filename, XYSet * set)
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
                    if (load_spec(line, &spec_) != 0)
                    {
                        fprintf(stderr, "load spec failed:\n\"%s\"\n", line);
                        return -1;
                    }
                    loaded_spec = 1;
                }
                else
                {
                    XY xy;
                    if (load_x(line, &xy) != 0)
                    {
                        fprintf(stderr, "parse line failed:\n\"%s\"\n", line);
                        bad_lines++;
                    }
                    total_lines++;
                    set->add(xy);
                }
            }
        }

        printf("loaded spec: %d colunms\n", (int)spec_.get_xtype_size());
        printf("loaded %d training samples, %d bad training samples\n", total_lines, bad_lines);

        set->spec() = spec_;
        return 0;
    }
};

int load_liblinear(const char * filename, XYSet * set)
{
    LibLinearLoader loader;
    set->clear();
    return loader.load(filename, set);
}

int load_gbdt(const char * filename, XYSet * set)
{
    GBDTLoader loader;
    set->clear();
    return loader.load(filename, set);
}
