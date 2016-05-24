#include "teca_tc_trajectory.h"

#include "teca_database.h"
#include "teca_table.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"

#include "teca_distance_function.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

namespace internal
{
void get_step_offsets(const long *step_ids, unsigned long n_rows,
  unsigned long &n_steps, std::vector<unsigned long> &step_counts,
  std::vector<unsigned long> &step_offsets)
{
    // count unique number of steps
    n_steps = 1;
    unsigned long n_m1 = n_rows - 1;
    for (unsigned long i = 0; i < n_m1; ++i)
    {
        if (step_ids[i] != step_ids[i+1])
            ++n_steps;
    }

    // compute num storms in each step
    step_counts.resize(n_steps);
    unsigned long q = 0;
    for (unsigned long i = 0; i < n_steps; ++i)
    {
        step_counts[i] = 1;
        while ((q < n_m1) && (step_ids[q] == step_ids[q+1]))
        {
          ++step_counts[i];
          ++q;
        }
        ++q;
    }

    // compute the offset to the first storm in each step
    step_offsets.resize(n_steps);
    step_offsets[0] = 0;
    for (unsigned long i = 1; i < n_steps; ++i)
        step_offsets[i] = step_offsets[i-1] + step_counts[i-1];
}

template<typename coord_t, typename var_t>
int teca_tc_trajectory(
    var_t r_crit, var_t wind_crit, double n_wind_crit,
    const long *time_step, const double *time, const int *storm_uid,
    const coord_t *d_lon, const coord_t *d_lat, const var_t *wind_max,
    const var_t *vort_max, const var_t *psl, const int *have_twc,
    const int *have_thick, const var_t *twc_max, const var_t *thick_max,
    unsigned long n_rows, p_teca_table track_table)
{
    const coord_t DEG_TO_RAD = M_PI/180.0;

    unsigned long track_id = 0;

    // convert from dsegrees to radians
    unsigned long nbytes = n_rows*sizeof(coord_t);
    coord_t *r_lon = static_cast<coord_t*>(malloc(nbytes));
    for (unsigned long i = 0; i < n_rows; ++i)
        r_lon[i] = DEG_TO_RAD*d_lon[i];

    coord_t *r_lat = static_cast<coord_t*>(malloc(nbytes));
    for (unsigned long i = 0; i < n_rows; ++i)
        r_lat[i] = DEG_TO_RAD*d_lat[i];

    bool *available = static_cast<bool*>(malloc(n_rows*sizeof(bool)));
    for (unsigned long i = 0; i < n_rows; ++i)
        available[i] = true;

    // build offset to time step table
    unsigned long n_steps;
    std::vector<unsigned long> step_counts;
    std::vector<unsigned long> step_offsets;

    internal::get_step_offsets(time_step, n_rows,
        n_steps, step_counts, step_offsets);

    // build the track start queue.
    // consider all tracks eminating from all storms.
    // as tracks form these are marked as used
    unsigned long n_m1 = n_rows - 1;
    std::vector<unsigned long> track_starts(n_rows);
    for (unsigned long i = 0; i < n_rows; ++i)
        track_starts[n_m1-i] = i;

    while (track_starts.size())
    {
        // for each potential track start
        unsigned long track_start = track_starts.back();
        track_starts.pop_back();

        if (available[track_start])
        {
            // this storm is not part of another track is now considered
            // whether or not it works out it is no longer available for
            // use in other tracks
            available[track_start] = false;

            // start the new track
            unsigned long max_track_len = n_steps - time_step[track_start];

            std::vector<unsigned long> new_track;
            new_track.reserve(max_track_len);
            new_track.push_back(track_start);

            double duration = 0.0;
            double wind_duration = 0.0;

            // now walk forward in time examining each storm in the
            // next time step.active step is next one forward in time
            unsigned long active_step = time_step[track_start] + 1;
            for (unsigned long j = active_step; j < n_steps; ++j)
            {
                // get position of the end of the track
                unsigned long track_tip = new_track.back();

                coord_t lon_0 = r_lon[track_tip];
                coord_t lat_0 = r_lat[track_tip];

                double t_0 = time[track_tip];
                double t_i = time[step_offsets[j]];
                double dt = t_i - t_0;

                // record duration.
                duration += dt;

                // apply storm duration criteria
                if ((wind_max[track_tip] >= wind_crit) &&
                  have_twc[track_tip] && have_thick[track_tip])
                  wind_duration += dt;

                // find the closest storm
                // note that r_crit is specified in km per day
                unsigned long closest_storm_id = 0;
                coord_t closest_storm_dist = r_crit*static_cast<coord_t>(dt);
                bool success = false;

                unsigned long n_storms = step_counts[j] - 1;
                for (unsigned int i = 0; i < n_storms; ++i)
                {
                    // check the storms distance. check them all since
                    // we need the closest
                    unsigned long storm_id = step_offsets[j] + i;

                    // skip storms that are already part of other tracks
                    if (!available[storm_id])
                      continue;

                    coord_t lon_i = r_lon[storm_id];
                    coord_t lat_i = r_lat[storm_id];

                    // compute the distance to the track tip
                    coord_t storm_dist =
                      teca_distance(lon_0, lat_0, lon_i, lat_i);

                    if (storm_dist <= closest_storm_dist)
                    {
                        // found one that's at least as close as
                        closest_storm_id = storm_id;
                        closest_storm_dist = storm_dist;
                        success = true;
                    }
                }

                if (success)
                {
                    // we were able to extend this track
                    new_track.push_back(closest_storm_id);
                    available[closest_storm_id] = false;
                }
                else
                {
                  // track ends here
                  break;
                }
            }

            // one track has been completed
            unsigned long track_len = new_track.size();

            if ((track_len > 1) && (wind_duration > n_wind_crit))
            {
                // output trajectory info
                //call teca_vector_long_write(new_track, 'track', error_unit)

                for (unsigned long i = 0; i < track_len; ++i)
                {
                    // output trajectory info
                    unsigned long storm_id = new_track[i];

                    track_table << track_id << storm_uid[storm_id]
                        << time_step[storm_id] << time[storm_id] << d_lon[storm_id]
                        << d_lat[storm_id] << duration << wind_duration << psl[storm_id]
                        << wind_max[storm_id] << vort_max[storm_id] << have_twc[storm_id]
                        << have_thick[storm_id] << twc_max[storm_id] << thick_max[storm_id];
                }

                ++track_id;
            }

            // free up memory for this track
            new_track.clear();
        }
    }

    // free up memory
    track_starts.clear();
    free(r_lon);
    free(r_lat);
    free(available);

    return 0;
}
};


// --------------------------------------------------------------------------
teca_tc_trajectory::teca_tc_trajectory() :
    max_daily_distance(900.0),
    min_wind_speed(17.0),
    min_wind_duration(2.0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_tc_trajectory::~teca_tc_trajectory()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_trajectory::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_tc_trajectory)");

    opts.add_options()
        TECA_POPTS_GET(double, prefix, max_daily_distance,
            "max distance a storm can move on the same track in single day")
        TECA_POPTS_GET(double, prefix, min_wind_speed,
            "minimum wind speed to be worthy of tracking")
        TECA_POPTS_GET(double, prefix, min_wind_duration,
            "minimum number of, not necessarily consecutive, days thickness, "
            "core temp, and wind speed criteria must be satisfied")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_trajectory::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, double, prefix, max_daily_distance)
    TECA_POPTS_SET(opts, double, prefix, min_wind_speed)
    TECA_POPTS_SET(opts, double, prefix, min_wind_duration)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_tc_trajectory::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_tc_trajectory::get_output_metadata" << endl;
#endif
    (void)port;
    teca_metadata out_md(input_md[0]);
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_tc_trajectory::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_tc_trajectory::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    teca_metadata req(request);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_trajectory::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_tc_trajectory::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_table candidates =
        std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!candidates)
    {
        if (rank == 0)
        {
            TECA_ERROR("empty input or not a table")
        }
        return nullptr;
    }

    // validate the input table
    const char *req_cols[] = {
        "step", "time", "storm_id", "lon", "lat", "surface_wind",
        "850mb_vorticity", "sea_level_pressure", "have_core_temp",
        "have_thickness", "core_temp", "thickness"};

    size_t n_req_cols = sizeof(req_cols)/sizeof(char*);
    for (size_t i = 0; i < n_req_cols; ++i)
    {
        if (!candidates->has_column(req_cols[i]))
        {
            TECA_ERROR("Candidate table missing \"" << req_cols[i] << "\"")
            return nullptr;
        }
    }

    // get the candidate storm properties
    const long *p_step =
        dynamic_cast<const teca_variant_array_impl<long>*>(
        candidates->get_column("step").get())->get();

    const double *p_time =
        dynamic_cast<const teca_variant_array_impl<double>*>(
        candidates->get_column("time").get())->get();

    const int *p_storm_id =
        dynamic_cast<const teca_variant_array_impl<int>*>(
        candidates->get_column("storm_id").get())->get();

    const_p_teca_variant_array lon = candidates->get_column("lon");
    const_p_teca_variant_array lat = candidates->get_column("lat");

    const_p_teca_variant_array wind_max =
        candidates->get_column("surface_wind");

    const_p_teca_variant_array vort_max =
        candidates->get_column("850mb_vorticity");

    const_p_teca_variant_array psl_min =
        candidates->get_column("sea_level_pressure");

    const int *p_have_twc =
        dynamic_cast<const teca_variant_array_impl<int>*>(
        candidates->get_column("have_core_temp").get())->get();

    const int *p_have_thick =
        dynamic_cast<const teca_variant_array_impl<int>*>(
        candidates->get_column("have_thickness").get())->get();

    const_p_teca_variant_array twc_max =
        candidates->get_column("core_temp");

    const_p_teca_variant_array thick_max =
        candidates->get_column("thickness");

    // create the table to hold storm tracks
    p_teca_table storm_tracks = teca_table::New();
    storm_tracks->copy_metadata(candidates);

    std::string time_units;
    storm_tracks->get_time_units(time_units);
    if (time_units.find("days since") == std::string::npos)
    {
        TECA_ERROR("Conversion for \"" << time_units << "\" not implemented")
        return nullptr;
    }

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        lon.get(), _COORD,

        const NT_COORD *p_lon =
            static_cast<const TT_COORD*>(lon.get())->get();

        const NT_COORD *p_lat =
            static_cast<const TT_COORD*>(lat.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            wind_max.get(), _VAR,

            // configure the output
            storm_tracks->declare_columns("track_id", int(),
                "storm_id", int(), "step", long(), "time", double(),
                "lon", NT_COORD(), "lat", NT_COORD(), "duration", double(),
                "wind_duration", double(), "surface_wind", NT_VAR(),
                "850mb_vorticity", NT_VAR(), "sea_level_pressure", NT_VAR(),
                "have_core_temp", int(), "have_thickness", int(),
                "core_temp", NT_VAR(), "thickness", NT_VAR());

            const NT_VAR *p_wind_max =
                dynamic_cast<const TT_VAR*>(wind_max.get())->get();

            const NT_VAR *p_vort_max =
                dynamic_cast<const TT_VAR*>(vort_max.get())->get();

            const NT_VAR *p_psl_min =
                dynamic_cast<const TT_VAR*>(psl_min.get())->get();

            const NT_VAR *p_twc_max =
                dynamic_cast<const TT_VAR*>(twc_max.get())->get();

            const NT_VAR *p_thick_max =
                dynamic_cast<const TT_VAR*>(thick_max.get())->get();

            // invoke the track finder
            if (internal::teca_tc_trajectory(
                static_cast<NT_VAR>(this->max_daily_distance),
                static_cast<NT_VAR>(this->min_wind_speed), this->min_wind_duration,
                p_step, p_time, p_storm_id, p_lon, p_lat, p_wind_max, p_vort_max,
                p_psl_min, p_have_twc, p_have_thick, p_twc_max, p_thick_max,
                candidates->get_number_of_rows(), storm_tracks))
            {
                TECA_ERROR("GFDL TC trajectory analysis encountered an error")
                return nullptr;
            }
            )
        )

    storm_tracks->to_stream(cerr);

    return storm_tracks;
}
