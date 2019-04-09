"""
Anki Vector Environment class definition
"""

import random
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import numpy as np
import cv2
from PIL import Image
import gym
from gym import error, spaces
import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees
from anki_vector.exceptions import VectorUnavailableException
import colorsys

from Mask_RCNN.run_mask_rcnn import *

IMAGE_DIM_INPUT = (360 / 2, 640 / 2)

ALL_POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveBack',
    'MoveRight',
    'MoveLeft',
    'LookUp',
    'LookDown',
    'RotateRight',
    'RotateLeft',
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
                                  255,
                                  #image[:, :, c] *
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

class AnkiVectorEnv(gym.Env):
    """
    Environment base class
    If the robot has low battery, it will get interrupted to go to the charging station and then
    restart a new episode
    TODO: check behaviour when falling
    TODO: check if resume after charging
    """
    def __init__(self, render=False):
        self.robot = anki_vector.Robot(enable_camera_feed=True)
        self.robot.connect()
        # TODO: use cube as landmark for the relative position where to start the experiment
        #self.robot.say_text("Environment ready")

        self.robot.behavior.set_lift_height(0.0)
        self.robot.behavior.set_head_angle(degrees(-.0))

        self.render = render
        self.degrees_rotate = 10
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        reward, done, info = 0, False, {}

        if action == 0:
            print('Turning left')
            self.robot.behavior.turn_in_place(degrees(self.degrees_rotate))
        elif action == 1:
            print('Turning right')
            self.robot.behavior.turn_in_place(degrees(-self.degrees_rotate))

        raw_state = self.robot.camera.latest_image  # todo could wait till action is completed above
        numpy_state = self.get_numpy_image(raw_state, resize=True, )

        #import pdb;pdb.set_trace()

        mask_rcnn_start_time = time.time()
        result = get_mask_rcnn_results(numpy_state)
        print('Mask RCNN time taken: {}'.format(round(time.time() - mask_rcnn_start_time, 2)))
        segmented_image = create_segmented_state(numpy_state, result)

        if self.render:
            cv2.imshow('Current raw image', numpy_state)
            cv2.imshow('Segmented image', segmented_image)
            cv2.waitKey(1)

        return numpy_state, reward, done, info

    def reset(self):
        self.robot.say_text("Reset Environment")
        # TODO: reset steps
        self.robot.say_text("Done")

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.disconnect()

    def get_numpy_image(self, image, resize=False, size=IMAGE_DIM_INPUT, return_PIL=False):
        """

        :param resize: whether to resize using PIL
        :param size: tuple e.g. (160, 80)
        :param return_PIL: whether to return PIL image
        :return: numpy array raw image in shape (240, 320, 3)
        """
        # if _display_debug_annotations != DEBUG_ANNOTATIONS_DISABLED:
        #     image = image.annotate_image(scale=2)
        # else:
        #image = image.raw_image
        if resize:
            image.thumbnail(size, Image.ANTIALIAS)
        if return_PIL:
            return image
        return np.asarray(image)

if __name__ == "__main__":
    env = AnkiVectorEnv(render=True)
    for episode_num in range(5):
        for step in range(100):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)