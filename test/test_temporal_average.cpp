#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_temporal_average.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_dataset_diff.h"
#include "teca_dataset_capture.h"
#include "teca_dataset_source.h"
#include "teca_index_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_mpi.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

// example use
// ./test/test_cf_reader ~/work/teca/data/'cam5_1_amip_run2.cam2.h2.1991-10-.*' tmp 0 -1 PS

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    if (argc < 5)
    {
        cerr << endl << "Usage error:" << endl
            << "test_cf_reader [input regex] [output] [first step] [last step] [filter width] [array] [array] ..." << endl
            << endl;
        return -1;
    }

    // parse command line
    string regex = argv[1];
    string baseline = argv[2];

    int have_baseline = 0;
    if (teca_file_util::file_exists(baseline.c_str()))
        have_baseline = 1;
    
    int filter_width = atoi(argv[3]);
    vector<string> arrays;
    arrays.push_back(argv[4]);
    for (int i = 5; i < argc; ++i)
        arrays.push_back(argv[i]);

    // create the cf reader
    p_teca_cf_reader r = teca_cf_reader::New();
    r->set_files_regex(regex);

    p_teca_normalize_coordinates c = teca_normalize_coordinates::New();
    c->set_input_connection(r->get_output_port());

    p_teca_temporal_average a = teca_temporal_average::New();
    a->set_filter_width(filter_width);
    a->set_filter_type(teca_temporal_average::backward);
    a->set_input_connection(c->get_output_port());

    //// Trying something
    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(a->get_output_port());

    // regression test
    if (have_baseline)
    {
        // run the test
        p_teca_cartesian_mesh_reader rea = teca_cartesian_mesh_reader::New();
        rea->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, rea->get_output_port());
        diff->set_input_connection(1, a->get_output_port());

        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;

        p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
        wri->set_input_connection(a->get_output_port());
        wri->set_file_name(baseline.c_str());

        wri->update();
    }

    return 0;
}
