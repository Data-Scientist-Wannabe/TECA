#ifndef teca_algorithm_util_h
#define teca_algorithm_util_h

#include <cstddef>

namespace teca_algorithm_util
{
// expand a dimension like 'x', 'y' or 'z' coordinates. If the dimension
// length is more than 2 entries then we are going to append values
// spaced equally from the end values. Which side to expand is determined
// by specifying either 'low' or 'high' arguments. If the dimension length
// is less than 2 values, then we are going to expand it with unit spaced
// values - 0, 1, 2, 3, etc.
template <typename num_t>
void expand_dimension(
    num_t *output_dim, const num_t *input_dim,
    size_t n_orig, size_t low, size_t high)
{
    if (n_orig >= 2)
    {
        num_t first_dim_val = input_dim[0];
        num_t last_dim_val = input_dim[n_orig-1];

        unsigned int diff_low = input_dim[1] - input_dim[0];
        unsigned int diff_high = input_dim[n_orig-1] - input_dim[n_orig-2];

        for (size_t i = 0; i < low; ++i)
        {
            output_dim[i] = first_dim_val - (low-i)*diff_low;
        }

        for (size_t i = 0; i < n_orig; ++i)
        {
            output_dim[low + i] = input_dim[i];
        }

        size_t ii = low + n_orig;
        for (size_t i = 0; i < high; ++i)
        {
            output_dim[ii + i] = last_dim_val + (i+1)*diff_high;
        }
    }
    else
    {
        size_t n_total = low + n_orig + high;

        for (size_t i = 0; i < n_total; ++i)
        {
            output_dim[i] = (num_t) i;
        }
    }
}
};
#endif
