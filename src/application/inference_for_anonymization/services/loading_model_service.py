import os
from application.inference.services.model_utils_service import check_model
from openvino.inference_engine import IECore


class LoadingModel:
    def __init__(self):
        self.path_model = self.path = '/models/'
        self.ie = IECore()

    def load_model(self, model_name):
        """
        load a model into the network so it can be used later on in the segmentation
        :param model_name:
        :return:
        """
        # Plugin initialization for specified device and load extensions library if specified.
        ie = self.ie
        modelname = self.path_model + model_name + '/' + model_name + '.xml'  # model_name
        net = ie.read_network(modelname, os.path.splitext(modelname)[0] + '.bin')
        image_input, image_info_input, (n, c, h, w), postprocessor = check_model(net)
        exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)
        return exec_net, image_input, image_info_input, (n, c, h, w), postprocessor
