import sys
import teca_py
import numpy as np

class teca_model_segmentation:
    """
    Computes summary statistics, histograms on sorted, classified,
    TC trajectory output.
    """
    @staticmethod
    def New():
        return teca_model_segmentation()

    def __init__(self):
        self.variable_name = "IVT"
        self.model_pt_path = None

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_tc_activity_execute(self))

    def __str__(self):
        return 'basename=%s' % (self.model_pt_path)

    def set_variable_name(self, var_name):
        """
        set the variable name that will be inputed to the segmentation model
        """
        self.variable_name = var_name

    def set_model_pt_path(self, model_pt_path):
        """
        set file path of the pretrained model that will be loaded by Pytorch
        """
        self.model_pt_path = model_pt_path

    def set_input_connection(self, obj):
        """
        set the input
        """
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        """
        get the output
        """
        return self.impl.get_output_port()

    def update(self):
        """
        execute the pipeline from this algorithm up.
        """
        self.impl.update()

    @staticmethod
    def get_predictions_execute(state):
        """
        return a teca_algorithm::execute function. a closure
        is used to gain state.
        """
        def execute(port, data_in, req):
            """
            expects an array of an input variable to run through
            the torch model and get the segmentation results as an 
            output.
            """
            in_mesh = teca.as_teca_cartesian_mesh(data_in[0])

            if in_mesh is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: empty input, or not a mesh\n')
                return teca_cartesian_mesh.New()

            import torch
            import torch.nn.functional as F

            if self.model_pt_path is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: pretrained model has not been specified\n')
                return teca_cartesian_mesh.New()

            model = None
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = torch.load(self.model_pt_path).cuda()
            else:
                device = torch.device("cpu")
                model = torch.load(self.model_pt_path)
            model.eval()

            lon = np.array(in_mesh.get_x_coordinates())
            lat = np.array(in_mesh.get_y_coordinates())

            arrays = in_mesh.get_point_arrays()

            var_array = arrays[self.variable_name]
            var_array = np.reshape(var_array, [len(lat), len(lon)])
            var_array = torch.from_numpy(var_array).to(device)

            # Disabling gradient calculation for efficiency
            # as backpropagation won't be called
            with torch.no_grad():
                pred = F.sigmoid(model(var_array))

            return in_mesh
        return execute
