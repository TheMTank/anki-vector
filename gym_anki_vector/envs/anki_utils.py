import random
import time

import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from matplotlib.ticker import NullLocator
from skimage.measure import find_contours
import numpy as np
import cv2

from PIL import Image
import gym
from gym import error, spaces
import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.exceptions import VectorUnavailableException
import colorsys

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

classes = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * 255,
                                  #(1 - alpha) + alpha * color[c] * 255,
                                  #255,  # white only for now
                                  image[:, :, c])
    return image

#boxes, masks, class_ids, class_names,
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            #class_names, r['scores'])

def create_segmented_state(image, result):
    """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """

    boxes = result['rois']
    masks = result['masks']
    class_ids = result['class_ids']
    #class_names
    scores = result['scores']
    ax = None
    figsize = (16, 16)
    colors = None
    title = None
    show_bbox = False
    show_mask = True
    # Number of instances
    N = boxes.shape[0]
    #import pdb;pdb.set_trace()
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    #masked_image = image.astype(np.uint32).copy()
    masked_image = np.zeros_like(image)
    for i in range(N):
        color = colors[i]

        # Bounding box
        '''if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)'''

        # Label
        '''if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")'''

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

    # todo return bbox area and masked_image and use right classes
    #cups_detected = class_ids[class_ids == 42]

    return masked_image

def create_segmented_bbox_image_and_reward(raw_state, detections, get_bbox_dense_reward=True):
    image = np.zeros_like(raw_state)
    reward = 0

    if detections and detections[0] is not None:
        detections = detections[0]  # list because of batch of images but only 1 image here

        if len(detections.size()) > 1:
            cup_detections = detections[detections[:, -1].type(torch.IntTensor) == 41]
            # if len(cup_detections.size()) > 1:
            if cup_detections.size(0) > 0:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = cup_detections[0]  # todo two cups?

                image[int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item()), :] = 255

                if get_bbox_dense_reward:
                    area = (y2 - y1) * (x2 - x1)
                    full_image_area = raw_state.shape[0] * raw_state.shape[1]
                    reward = area / full_image_area  # reward is normalised bbox area according to max area

    return image, reward

def create_image_with_bounding_boxes(raw_state, detections, img_size):
    """

    :param raw_state:
    :param detections:
    :param img_size: must be square image
    :param classes
    :return:
    """
    detections = detections[0]  # list because of batch of images but only 1 image here
    # Create plot
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(raw_state)

    # The amount of padding that was added
    pad_x = max(raw_state.shape[0] - raw_state.shape[1], 0) * (img_size / max(raw_state.shape))
    pad_y = max(raw_state.shape[1] - raw_state.shape[0], 0) * (img_size / max(raw_state.shape))
    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw bounding boxes and labels of detections
    #import pdb;pdb.set_trace()
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            # print('projected coordinates: x1, y1, x2, y2: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x1, y1, x2, y2))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * raw_state.shape[0]
            box_w = ((x2 - x1) / unpad_w) * raw_state.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * raw_state.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * raw_state.shape[1]

            # print('original coordinates: x1, y1, x2, y2: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x1, y1, x1 + box_w, y1 + box_h))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    fig.canvas.draw()

    # Now we can save it to a numpy array. todo think
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # import pdb;pdb.set_trace()
    return data