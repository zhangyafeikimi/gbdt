#include "lm.h"
#include "lm-scorer.h"
#include "lm-util.h"
#include <stdio.h>
#include <functional>

int main()
{
    std::vector<double> unsorted;
    std::vector<size_t> indices;

    unsorted.push_back(1.0);
    unsorted.push_back(5.0);
    unsorted.push_back(3.0);
    unsorted.push_back(2.0);
    unsorted.push_back(4.0);
    unsorted.push_back(10.0);

    sort_indices(unsorted, &indices, std::greater_equal<double>());

    for (size_t i=0; i<indices.size(); i++)
        printf("%f\n", unsorted[indices[i]]);

    return 0;
}
