import numpy as np
import cv2
import logging as log

from PIL import Image
from application.inference.services import monitors
from application.inference.services.tracker_service import StaticIOUTracker
from application.inference.services.visualizer_service import Visualizer
from application.fetch_labels.fetch_labels import FetchLabels


class ImageVisualizer:

    def __init__(self):
        self.fetch_labels = FetchLabels()

    def visualizer(self, input_data, model_name):
        """
        check if the image is loaded in opencv, get all the labels and visualize the result
        :param input_data:
        :param model_name:
        :return: cap, visualizer, tracker, presenter
        """
        imput = Image.open(input_data.file, "r").convert('RGB')
        input_im = np.array(imput)
        cv2.imwrite(input_data.filename, input_im)
        imput = input_data.filename
        try:
            input_source = int(imput)
        except ValueError:
            input_source = imput
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            log.error('Failed to open "{}"'.format(imput))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        tracker = StaticIOUTracker()
        class_labels = self.fetch_labels.get_labels(model_name)
        presenter = monitors.Presenter(1, 45,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))
        visualizer = Visualizer(class_labels, show_boxes=False, show_scores=False)
        return cap, visualizer, tracker, presenter
