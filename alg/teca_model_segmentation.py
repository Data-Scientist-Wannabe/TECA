import sys
import teca_py
import numpy as np
import torch
import torch.nn.functional as F

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
        self.pred_name = self.variable_name + "_PRED"
        self.model_pt_path = None
        self.model = None
        self.device = None
        self.is_cuda = True

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_tc_activity_execute(self))

    def __str__(self):
        ms_str = 'variable_name=%s, pred_name=%d, model_pt_path=%s, is_cuda=%s\n\n'%( \
            self.variable_name, self.pred_name, self.model_pt_path, str(self.is_cuda))

        ms_str += 'model:\n%s'%(str(self.model))

        return ms_str

    def set_variable_name(self, name):
        """
        set the variable name that will be inputed to the segmentation model
        """
        self.variable_name = name

    def set_pred_name(self, name):
        """
        set the variable name that will be inputed to the segmentation model
        """
        self.pred_name = name

    def set_torch_device(self, is_cuda=True):
        """
        Set to True or False to choose Pytorch's device
        (True for cuda or False for cpu)
        """
        if not is_cuda:
            self.is_cuda = False

    def set_model_pt_path(self, model_pt_path):
        """
        set file path of the pretrained model that will be loaded by Pytorch
        """
        self.model_pt_path = model_pt_path

        if self.is_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = torch.load(self.model_pt_path).cuda()
        else:
            self.device = torch.device("cpu")
            self.model = torch.load(self.model_pt_path)
        self.model.eval()

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

    def get_report_callback(self):
        """
        return a teca_algorithm::report function adding the output name
        that will hold the output predictions of the used model.
        """
        def report(port, rep_in):
            rep = teca_metadata(rep_in)

            if not rep['variables']:
                rep['variables'] = []

            if self.pred_name:
                rep['variables'].append(self.pred_name)

            return rep
        return report


    def get_request_callback(self):
        """
        return a teca_algorithm::request function adding the variable name
        that the pretrained model will process.
        """
        def request(port, md_in, req_in):
            req = teca_metadata(req_in)

            if not req['arrays']:
                req['arrays'] = []
            
            if self.variable_name:
                req['arrays'].append(self.variable_name)

            return [req]
        return request

    def get_predictions_execute(self):
        """
        return a teca_algorithm::execute function
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


            if self.model_pt_path is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: pretrained model has not been specified\n')
                return teca_cartesian_mesh.New()

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

            if pred is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: Model failed to get predictions\n')
                return teca_cartesian_mesh.New()

            out_mesh.shallow_copy(in_mesh)

            out_mesh.get_point_arrays().set(self.pred_name, pred)

            return out_mesh
        return execute
