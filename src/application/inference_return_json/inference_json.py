import os
import numpy as np
import cv2
import json

from application.inference_return_json.services.loading_model_service import LoadingModel
from application.inference_return_json.services.image_visualizer_service import ImageVisualizer
from application.fetch_labels.fetch_labels import FetchLabels


class Inference:
    def __init__(self):
        self.image_visualizer = ImageVisualizer()
        self.model_loading = LoadingModel()
        self.prediction = {}
        self.path = './'
        self.fetch_labels = FetchLabels()
        file_name = self.path + 'final_json.json'
        file_exists = os.path.exists(file_name)

        if file_exists:
            file = 'final_json.json'
            path = os.path.join(self.path, file)
            os.remove(path)
        else:
            self.full_path = file_name

    def image_inference(self, model_name: str, input_data):
        """
        take model and picture from the user and return json file
        :param model_name: model name
        :param input_data: image
        :return:
        """
        exec_net, image_input, image_info_input, (n, c, h, w), postprocessor = self.model_loading.load_model(model_name)
        cap, visualizer, tracker, presenter = self.image_visualizer.visualizer(input_data,model_name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

            input_image_size = input_image.shape[:2]
            input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                               (0, w - input_image_size[1]),
                                               (0, 0)),
                                 mode='constant', constant_values=0)
            # Change data layout from HWC to CHW.
            input_image = input_image.transpose((2, 0, 1))
            input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
            input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)
            # Run the net.
            feed_dict = {image_input: input_image}
            if image_info_input:
                feed_dict[image_info_input] = input_image_info
            outputs = exec_net.infer(feed_dict)
            # Parse detection results of the current request
            scores, classes, boxes, masks = postprocessor(
                outputs, scale_x, scale_y, *frame.shape[:2], h, w, 0.5)
            os.remove(input_data.filename)
            class_labels = self.fetch_labels.get_labels(model_name)

            t = 0
            for key2 in [class_labels[i] for i in classes]:
                x1 = str(boxes[t][0])
                y1 = str(boxes[t][1])
                x2 = str(boxes[t][2])
                y2 = str(boxes[t][3])

                if key2 in self.prediction.keys():
                    value_init = self.prediction.get(key2)
                    self.prediction[key2] = x1, y1, x2, y2
                    value = value_init, self.prediction.get(key2)
                    self.prediction[key2] = value

                else:
                    self.prediction[key2] = x1, y1, x2, y2

                t = t + 1

            with open('./final_json.json', 'w') as file:
                json.dump(self.prediction, file)

            with open('./final_json.json','r') as file:
                json_object = json.load(file)

            return json_object
        cv2.destroyAllWindows()
        cap.release()