"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np


class Visualizer(object):

    def __init__(self, class_labels, confidence_threshold=0.5, show_boxes=False,
                 show_masks=True, show_scores=False):
        super().__init__()
        self.class_labels = class_labels
        self.confidence_threshold = confidence_threshold
        self.class_color_palette = np.asarray([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.show_masks = show_masks
        self.show_boxes = show_boxes
        self.show_scores = show_scores

    def __call__(self, image, boxes, classes, scores, presenter, segms=None, ids=None):
        result = image.copy()

        # Filter out detections with low confidence.
        filter_mask = scores > self.confidence_threshold
        scores = scores[filter_mask]
        classes = classes[filter_mask]
        boxes = boxes[filter_mask]

        presenter.drawGraphs(result)

        if self.show_masks and segms is not None:
            segms = list(segm for segm, show in zip(segms, filter_mask) if show)
            result = self.overlay_masks(result, segms, classes, ids)

        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes)
        return result

    def compute_colors_for_labels(self, labels):
        colors = labels[:, None] * self.class_color_palette
        colors = (colors % 255).astype(np.uint8)
        return colors

    def overlay_boxes(self, image, boxes, classes):
        colors = self.compute_colors_for_labels(classes).tolist()
        for box, color in zip(boxes, colors):
            box = box.astype(int)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
        return image

    def overlay_masks(self, image, masks, classes, ids=None):
        colors = self.compute_colors_for_labels(classes).tolist()
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        black = np.zeros(3, dtype=np.uint8)
        
        for i, (mask, color) in enumerate(zip(masks, colors)):

            mask_color = classes[i]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),
                           dst=aggregated_colored_mask, mask=mask)
        return aggregated_colored_mask

    def overlay_class_names(self, image, boxes, classes, scores, show_score=True):
        labels = [self.class_labels[i] for i in classes]
        template = '{}: {:.2f}' if show_score else '{}'
        white = (255, 255, 255)

        for box, score, label in zip(boxes, scores, labels):
            s = template.format(label, score)
            textsize = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(int)
            cv2.putText(image, s, tuple(position), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)

        return image
