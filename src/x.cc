#include "x.h"

FILE * yfopen(const char * filename, const char * mode)
{
    FILE * fp = fopen(filename, mode);
    if (fp == 0)
        fprintf(stderr, "open \"%s\" failed\n", filename);
    return fp;
}

FILE * xfopen(const char * filename, const char * mode)
{
    FILE * fp = fopen(filename, mode);
    if (fp == 0)
    {
        fprintf(stderr, "open \"%s\" failed\n", filename);
        exit(1);
    }
    return fp;
}

void * xmalloc(size_t size)
{
    void * p = malloc(size);
    if (p == 0)
    {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return p;
}

void * xrealloc(void * memory, size_t new_size)
{
    void * p = realloc(memory, new_size);
    if (p == 0)
    {
        fprintf(stderr, "realloc failed\n");
        exit(1);
    }
    return p;
}

double xatof(const char * str)
{
    char * endptr;
    double d;

    errno = 0;
    d = strtod(str, &endptr);
    if (errno != 0 || str == endptr)
    {
        fprintf(stderr, "%s is not an double\n", str);
        exit(1);
    }
    return d;
}

int xatoi(const char *str)
{
    char * endptr;
    int i;

    errno = 0;
    i = (int) strtol(str, &endptr, 10);
    if (errno != 0 || str == endptr)
    {
        fprintf(stderr, "%s is not an integer\n", str);
        exit(1);
    }
    return i;
}
