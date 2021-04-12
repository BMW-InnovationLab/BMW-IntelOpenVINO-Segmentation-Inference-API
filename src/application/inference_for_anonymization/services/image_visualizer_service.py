import numpy as np
import cv2
import logging as log

from PIL import Image
from application.inference_for_anonymization.services import monitors
from application.inference_for_anonymization.services.tracker_service import StaticIOUTracker
from application.inference_for_anonymization.services.visualizer_service import Visualizer


class ImageVisualizer:

    def __init__(self):
        pass

    def visualizer(self, input_data):
        """
        check if the image is loaded in opencv, get all the labels and visualize the result
        :param input_data:
        :param model_name:
        :return: cap, visualizer, tracker, presenter
        """
        imput = Image.open(input_data.file, "r").convert('RGB')
        input_im = np.array(imput)
        cv2.imwrite('test.png',input_im)
        imput = 'test.png'
        try:
            input_source = int(imput)
        except ValueError:
            input_source = imput
        cap = cv2.VideoCapture(input_source)
        img2 = cv2.imread(input_source)
        if not cap.isOpened():
            log.error('Failed to open "{}"'.format(imput))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        tracker = StaticIOUTracker()
        presenter = monitors.Presenter(1, 45,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))
        visualizer = Visualizer(None, show_boxes=False, show_scores=False)
        return img2, visualizer, tracker, presenter
