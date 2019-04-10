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
#from skimage.measure import find_contours
#import numpy as np
import cv2
from PIL import Image
import gym
from gym import error, spaces
import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.exceptions import VectorUnavailableException
import colorsys

#from Mask_RCNN.run_mask_rcnn import *
from PyTorch_YOLOv3.detect_on_image import *  # todo find better way
#from PyTorch_YOLOv3 import detect_on_np_image

from anki_utils import create_image_with_bounding_boxes

#opt

#IMAGE_DIM_INPUT = (360 // 2, 640 // 2)
#IMAGE_DIM_INPUT = (640 // 2, 360 // 2)
IMAGE_DIM_INPUT = (416, 416)


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
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        reward, done, info = 0, False, {}

        if action == 0:
            print('Turning left')
            self.robot.behavior.turn_in_place(degrees(self.degrees_rotate))
        elif action == 1:
            print('Turning right')
            self.robot.behavior.turn_in_place(degrees(-self.degrees_rotate))
        elif action == 2:
            print('Going forward')
            self.robot.behavior.drive_straight(distance_mm(40), speed_mmps(50))

        raw_state = self.robot.camera.latest_image  # todo could wait till action is completed above
        numpy_state = self.get_numpy_image(raw_state, resize=True)

        #import pdb;pdb.set_trace()

        # Mask RCNN
        #mask_rcnn_start_time = time.time()
        #result = get_mask_rcnn_results(numpy_state)
        #print('Mask RCNN time taken: {}'.format(round(time.time() - mask_rcnn_start_time, 2)))
        #segmented_image = create_segmented_state(numpy_state, result)

        # YOLO
        detections = detect_on_np_image(numpy_state)
        #bbox_image = create_image_with_bounding_boxes(numpy_state, detections)


        if self.render:
            cv2.imshow('Current raw image', numpy_state)
            #cv2.imshow('Segmented image', segmented_image)
            #cv2.imshow('YOLO image', bbox_image)

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
        image = np.asarray(image)
        if resize:
            #image.thumbnail(size, Image.ANTIALIAS)
            image = cv2.resize(image, IMAGE_DIM_INPUT, interpolation=cv2.INTER_AREA)
        #if return_PIL:
        #    return image
        return image

if __name__ == "__main__":
    env = AnkiVectorEnv(render=True)
    for episode_num in range(5):
        for step in range(100):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)