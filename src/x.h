#ifndef GBDT_X_H
#define GBDT_X_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

FILE * yfopen(const char * filename, const char * mode);
FILE * xfopen(const char * filename, const char * mode);
void * xmalloc(size_t size);
void * xrealloc(void * memory, size_t new_size);
double xatof(const char *str);
int xatoi(const char *str);

class ScopedFile
{
private:
    FILE * px_;
    ScopedFile(const ScopedFile&);
    ScopedFile& operator=(const ScopedFile&);

public:
    explicit ScopedFile(FILE * p = 0) : px_(p) {}

    ~ScopedFile()
    {
        if (px_)
            fclose(px_);
    }
};

template <class T>
class ScopedPtrMalloc
{
private:
    T ptr_;
    ScopedPtrMalloc(const ScopedPtrMalloc&);
    ScopedPtrMalloc& operator=(const ScopedPtrMalloc&);

public:
    explicit ScopedPtrMalloc(T p = 0): ptr_(p) {}

    ~ScopedPtrMalloc()
    {
        if (ptr_)
            free((void*) ptr_);
    }

    void set_for_realloc(T p)
    {
        ptr_ = p;
    }
};

#endif// GBDT_X_H
