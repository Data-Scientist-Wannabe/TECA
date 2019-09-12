#ifndef teca_lat_lon_padding_h
#define teca_lat_lon_padding_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_lat_lon_padding)

/**
Pads the specified scalar field with zeroes or, if specified, pad_value.

note that user specified values take precedence over request keys. When using
request keys be sure to include the variable post-fix.
*/
class teca_lat_lon_padding : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_lat_lon_padding)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_lat_lon_padding)
    TECA_ALGORITHM_CLASS_NAME(teca_lat_lon_padding)
    ~teca_lat_lon_padding();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    TECA_ALGORITHM_PROPERTY(std::size_t, py_low)
    TECA_ALGORITHM_PROPERTY(std::size_t, py_high)

    TECA_ALGORITHM_PROPERTY(std::size_t, px_low)
    TECA_ALGORITHM_PROPERTY(std::size_t, px_high)

    // set the name of the array that the padding will apply on
    TECA_ALGORITHM_PROPERTY(std::string, field_to_pad)

    // a string to be appended to the name of the output variable
    // setting this to an empty string will result in the padded array
    // replacing the input array in the output. default is an empty
    // string ""
    TECA_ALGORITHM_PROPERTY(std::string, variable_post_fix)

protected:
    teca_lat_lon_padding();

    int get_field_to_pad(std::string &field_var);

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::size_t py_low, py_high;
    std::size_t px_low, px_high;
    std::string field_to_pad;
    std::string variable_post_fix;
};

#endif
