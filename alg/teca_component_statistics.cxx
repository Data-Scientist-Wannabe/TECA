#include "teca_component_statistics.h"

#include "teca_cartesian_mesh.h"
#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_component_statistics::teca_component_statistics()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_component_statistics::~teca_component_statistics()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_component_statistics::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    (void) prefix;

    options_description opts("Options for "
        + (prefix.empty()?"teca_component_statistics":prefix));

    /*opts.add_options()
        TECA_POPTS_GET(std::vector<std::string>, prefix, dependent_variables,
            "list of arrays to compute statistics for")
        ;*/

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_component_statistics::set_properties(
    const std::string &prefix, variables_map &opts)
{
    (void) prefix;
    (void) opts;

    //TECA_POPTS_SET(opts, std::vector<std::string>, prefix, dependent_variables)
}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_component_statistics::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_statistics::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs(1, request);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_component_statistics::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_component_statistics::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // get time metadta
    unsigned long s;
    in_mesh->get_time_step(s);

    double t;
    in_mesh->get_time(t);

    std::string calendar;
    in_mesh->get_calendar(calendar);

    std::string time_units;
    in_mesh->get_time_units(time_units);

    // get the component metadata
    teca_metadata dsmd = in_mesh->get_metadata();

    p_teca_variant_array component_ids = dsmd.get("component_ids");
    if (!component_ids)
    {
        TECA_ERROR("No component ids present")
        return nullptr;
    }

    unsigned long n_comps = component_ids->size();

    // make a time column
    p_teca_double_array time = teca_double_array::New();
    time->resize(n_comps);
    double *pt = time->get();
    for (unsigned long i = 0; i < n_comps; ++i)
        pt[i] = t;

    // make a time step column
    p_teca_unsigned_long_array step = teca_unsigned_long_array::New();
    step->resize(n_comps);
    unsigned long *ps = step->get();
    for (unsigned long i = 0; i < n_comps; ++i)
        ps[i] = s;

    // compute a glonbal id, from the time step and component id
    p_teca_unsigned_long_array global_component_ids =
        teca_unsigned_long_array::New();

    global_component_ids->resize(n_comps);
    unsigned long *pgcid = global_component_ids->get();
    TEMPLATE_DISPATCH_I(teca_variant_array_impl,
        component_ids.get(),
        NT *pcid = std::static_pointer_cast<TT>(component_ids)->get();
        unsigned long base = s*1000000;
        for (unsigned long i = 0; i < n_comps; ++i)
            pgcid[i] = base + pcid[i];
        )

    // get the area, if it has been computed
    p_teca_variant_array component_area = dsmd.get("component_area");

    // put all these columns int the output table
    p_teca_table table = teca_table::New();

    table->set_calendar(calendar);
    table->set_time_units(time_units);

    table->append_column("time", time);
    table->append_column("step", step);
    table->append_column("global_component_ids", global_component_ids);
    table->append_column("component_ids", component_ids);

    if (component_area)
        table->append_column("component_area", component_area);

    return table;
}
