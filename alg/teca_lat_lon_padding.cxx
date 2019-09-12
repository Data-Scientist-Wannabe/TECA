#include "teca_lat_lon_padding.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_metadata_util.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#include <complex.h>

using std::cerr;
using std::endl;

namespace {

// damp the input array using inverted gaussian
template <typename num_t>
void apply_padding(
    num_t *output, const num_t *input, size_t n_lat, size_t n_lon,
    size_t py_low, size_t px_low, size_t nx_new)
{
    size_t y_extent = py_low + n_lat; 
    size_t x_extent = px_low + n_lon;

    for (size_t j = py_low; j < y_extent; ++j)
    {
        size_t jj = (j - py_low) * n_lon;
        size_t jjj = j * nx_new;
        for (size_t i = px_low; i < x_extent; ++i)
        {
            output[jjj + i] = input[jj + i - px_low];
        }
    }
}

};

// --------------------------------------------------------------------------
teca_lat_lon_padding::teca_lat_lon_padding() :
    py_low(0), py_high(0),
    px_low(0), px_high(0),
    field_to_pad(""),
    variable_post_fix("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_lat_lon_padding::~teca_lat_lon_padding()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_lat_lon_padding::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_lat_lon_padding":prefix));
    
    opts.add_options()
        TECA_POPTS_GET(size_t, prefix, py_low,
            "set the y-dimenstion pad-low value. low means the negative "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, py_high,
            "set the y-dimenstion pad-high value. high means the positive "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, px_low,
            "set the x-dimenstion pad-low value. low means the negative "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, px_high,
            "set the x-dimenstion pad-high value. high means the positive "
            "side of the dimesnion")
        TECA_POPTS_GET(std::string, prefix, field_to_pad,
            "set the field that will be padded")
        TECA_POPTS_GET(std::string, prefix, variable_post_fix,
            "set the post-fix that will be attached to the field "
            "that will be saved in the output")
        ;
    
    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_lat_lon_padding::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, size_t, prefix, py_low)
    TECA_POPTS_SET(opts, size_t, prefix, py_high)
    TECA_POPTS_SET(opts, size_t, prefix, px_low)
    TECA_POPTS_SET(opts, size_t, prefix, px_high)
    TECA_POPTS_SET(opts, std::string, prefix, field_to_pad)
    TECA_POPTS_SET(opts, std::string, prefix, variable_post_fix)
}
#endif


// --------------------------------------------------------------------------
int teca_lat_lon_padding::get_field_to_pad(std::string &field_var)
{
    if (this->field_to_pad.empty())
        return -1;
    else
        field_var = this->field_to_pad;

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_lat_lon_padding::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_latitude_damper::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &field_var = this->field_to_pad;
    const std::string &var_post_fix = this->variable_post_fix;
    if (!field_var.empty() && !var_post_fix.empty())
    {
        out_md.append("variables", field_var + var_post_fix);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_lat_lon_padding::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_latitude_damper::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;
    teca_metadata req(request);

    // get the name of the field to request
    std::string field_var;
    if (this->get_field_to_pad(field_var))
    {
        TECA_ERROR("No field to pad specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(field_to_pad);

    // Cleaning off the postfix for arrays passed in the pipeline. 
    // For ex a down stream could request "foo_damped" then we'd
    // need to request "foo". also remove "foo_damped" from the
    // request.
    const std::string &var_post_fix = this->variable_post_fix;
    if (!var_post_fix.empty())
    {
        teca_metadata_util::remove_post_fix(arrays, var_post_fix);
    }

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_lat_lon_padding::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_latitude_damper::execute" << endl;
#endif

    (void)port;
    (void)request;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(in_mesh->new_instance());

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array names
    std::string field_var;
    if (this->get_field_to_pad(field_var))
    {
        TECA_ERROR("No field specified to pad")
        return nullptr;
    }

    // set the damped array in the output
    std::string out_var_name = field_var + this->variable_post_fix;

    // get the output metadata to add results to after the filter is applied
    teca_metadata &out_metadata = out_mesh->get_metadata();

    size_t py_low = this->get_py_low();
    size_t py_high = this->get_py_high();
    size_t px_low = this->get_px_low();
    size_t px_high = this->get_px_high();

    if (py_low || py_high || px_low || px_high)
    {
        const_p_teca_variant_array field_array = in_mesh->get_point_arrays()->get(field_var);
        if (!field_array)
        {
            TECA_ERROR("Field array \"" << field_var
                << "\" not present.")
            return nullptr;
        }

        // get the coordinate axes
        const_p_teca_variant_array lat = in_mesh->get_y_coordinates();
        const_p_teca_variant_array lon = in_mesh->get_x_coordinates();

        size_t n_lat = lat->size();
        size_t n_lon = lon->size();

        size_t ny_new = py_low + n_lat + py_high;
        size_t nx_new = px_low + n_lon + px_high;

        p_teca_variant_array padded_array = field_array->new_instance(ny_new * nx_new);

        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            padded_array.get(),

            const NT* p_field_array = static_cast<const TT*>(field_array.get())->get();
            NT* p_padded_array = static_cast<TT*>(padded_array.get())->get();

            memset(p_padded_array, 0, ny_new*nx_new*sizeof(NT));

            ::apply_padding(p_padded_array, p_field_array,
                            n_lat, n_lon, py_low, px_low, nx_new);

            out_mesh->get_point_arrays()->set(out_var_name, padded_array);
        )
    }

    out_metadata.set(out_var_name + "_py_low", _py_low);
    out_metadata.set(out_var_name + "_py_high", py_high);
    out_metadata.set(out_var_name + "_px_low", px_low);
    out_metadata.set(out_var_name + "_px_high", px_high);

    return out_mesh;
}

