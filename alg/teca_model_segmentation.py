import sys
import teca_py
import numpy as np
import torch
import torch.nn.functional as F

class teca_model_segmentation:
    """
    Given an input field of integrated vapor transport,
    calculates the probability of AR presence at each gridcell.
    """
    @staticmethod
    def New():
        return teca_model_segmentation()

    def __init__(self):
        self.variable_name = "IVT"
        self.pred_name = self.variable_name + "_PRED"
        self.transform_fn = None
        self.transport_fn_args = None
        self.model = None
        self.device = self.set_torch_device()

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_predictions_execute())

    def __str__(self):
        ms_str = 'variable_name=%s, pred_name=%d\n\n'%( \
            self.variable_name, self.pred_name)

        ms_str += 'model:\n%s\n\n'%(str(self.model))

        ms_str += 'device:\n%s\n'%(str(self.device))

        return ms_str

    def set_variable_name(self, name):
        """
        set the variable name that will be inputed to the model
        """
        self.variable_name = name

    def set_pred_name(self, name):
        """
        set the variable name that will be the output to the model
        """
        self.pred_name = name

    def set_transform_fn(self, fn, *args):
        """
        if the data need to be transformed in a way then a function
        could be provided to be applied on the requested data before
        running it to the model.
        """
        if not hasattr(fn, '__call__'):
            sys.stderr.write('ERROR: The provided data transform function \
                is not a function\n')
            return False

        if not args:
            sys.stderr.write('ERROR: The provided data transform function \
                must at least have 1 argument -- the data array object to \
                apply the transformation on.\n')
            return False

        self.transform_fn = fn
        self.transport_fn_args = args
        return True

    def set_torch_device(self, device="cpu"):
        """
        Set device to either 'cuda' or 'cpu'
        """
        if device == "cuda" and not torch.cuda.is_available():
            # TODO if this is part of a parallel pipeline then
            # only rank 0 should report an error.
            sys.stderr.write('ERROR: Couldn\'t set device to cuda, cuda is \
                not available\n')
            return torch.device('cpu')

        return torch.device(device)

    def set_model(self, model):
        """
        set Pytorch pretrained model 
        """
        self.model = model
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
            #print("report:")
            #print(type(rep_in))
            #print(rep_in)
            rep_temp = rep_in[0]
            #print(str(rep_temp))
            rep = teca_py.teca_metadata(rep_temp)

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
            if not self.variable_name:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: No variable to request specifed\n')
                return []

            #print("request:")
            #print(type(req_in))
            #print(req_in)
            #print(str(req_in))
            req = teca_py.teca_metadata(req_in)

            arrays = []
            if req.has('arrays'):
                arrays = req['arrays']
            
            arrays.append(self.variable_name)
            req['arrays'] = arrays

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
            print("execute:")
            print(type(data_in))
            #print(data_in[0])
            #data_in = data_in[0]
            in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

            if in_mesh is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: empty input, or not a mesh\n')
                return teca_py.teca_cartesian_mesh.New()

            if self.model is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: pretrained model has not been specified\n')
                return teca_py.teca_cartesian_mesh.New()

            lat = np.array(in_mesh.get_y_coordinates())
            lon = np.array(in_mesh.get_x_coordinates())

            arrays = in_mesh.get_point_arrays()

            var_array = arrays[self.variable_name]
            print("var_array.len: %s" % (str(len(var_array))))
            print("lat.len: %s" % (str(len(lat))))
            print("lon.len: %s" % (str(len(lon))))

            if self.transform_fn:
                var_array = self.transform_fn(var_array, *self.transport_fn_args)

            var_array = torch.from_numpy(var_array).to(self.device)

            with torch.no_grad():
                pred = F.sigmoid(self.model(var_array))

            if pred is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: Model failed to get predictions\n')
                return teca_py.teca_cartesian_mesh.New()
            
            out_mesh = teca_py.teca_cartesian_mesh.New()
            out_mesh.shallow_copy(in_mesh)
            print("pred:")
            print(type(pred))
            print(type(pred.numpy()))
            print(pred.shape)
            print(pred)
            print("pred.numpy().flatten().shape")
            print(pred.numpy().flatten().shape)

            pred = teca_py.teca_variant_array.New(pred.numpy().flatten())
            out_mesh.get_point_arrays().set(self.pred_name, pred)
            
            return out_mesh
        return execute
