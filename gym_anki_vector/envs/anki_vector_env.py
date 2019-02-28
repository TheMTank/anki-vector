"""
Base Environment class definition
"""
import functools
import threading
import time

import gym

import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees
from anki_vector.exceptions import VectorUnavailableException

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
    def __init__(self):
        self.robot = anki_vector.Robot(enable_camera_feed=True)
        self.robot.connect()
        # TODO: use cube as landmark for the relative position where to start the experiment
        self.robot.say_text("Environment ready")

    def step(self, action):
        reward, done, info = 0, False, {}
        state = self.robot.camera.latest_image
        print("Turn Vector in place...")
        self.robot.behavior.turn_in_place(degrees(360))
        print('Finished turn')

        return state, reward, done, info

    def reset(self):
        self.robot.say_text("Reset Environment")
        # TODO: reset steps
        self.robot.say_text("Done")

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.disconnect()


