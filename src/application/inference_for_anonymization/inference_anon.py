import os
import io
import numpy as np
import cv2
from PIL import Image



from application.inference_for_anonymization.services.loading_model_service import LoadingModel
from application.inference_for_anonymization.services.image_visualizer_service import ImageVisualizer

from application.fetch_palette.fetch_palette import FetchPalette

class Inference:
    def __init__(self):
        self.image_visualizer = ImageVisualizer()
        self.model_loading = LoadingModel()
        self.palette = []

    def image_inference(self, model_name: str, input_data):
        """
        take model and picture from the user and return the final segmentation image
        :param model_name: model name
        :param input_data: image
        :return:
        """
        exec_net, image_input, image_info_input, (n, c, h, w), postprocessor = self.model_loading.load_model(model_name)
        image, visualizer, tracker, presenter = self.image_visualizer.visualizer(input_data)
        # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
        scale_x = scale_y = min(h / image.shape[0], w / image.shape[1])
        input_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
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
                outputs, scale_x, scale_y, *image.shape[:2], h, w, 0.5)
        # Get instance track IDs.
        masks_tracks_ids = None
        if tracker is not None:
            masks_tracks_ids = tracker(masks, classes)
        # Visualize masks.
        frame = visualizer(image, boxes, classes, scores, presenter, masks, masks_tracks_ids)
        original=Image.fromarray(frame.astype('uint8')).convert("L")
        palette = FetchPalette().get_palette(model_name)
        original.putpalette(palette)
        converted = original.convert("P")
        converted.save("result.jpg","png")
        os.remove('test.png')