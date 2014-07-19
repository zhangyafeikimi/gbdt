#include "x.h"

FILE * yfopen(const char * filename, const char * mode)
{
    FILE * fp = fopen(filename, mode);
    if (fp == 0)
        fprintf(stderr, "open \"%s\" failed\n", filename);
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
