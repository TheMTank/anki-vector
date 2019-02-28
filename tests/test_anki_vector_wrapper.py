"""
Tests related to environment wrapper
"""
import time
import unittest

from gym_anki_vector.envs.anki_vector_env import AnkiVectorEnv


class TestAI2ThorEnv(unittest.TestCase):
    """
    General environment generation tests
    """
    def test_environments_runs(self):
        """
        Checks to see if the environment still runs and nothing breaks. Useful for continuous
        deployment and keeping master stable. Also, we check how much time 10 steps takes within
        the environment. Final assert checks if max_episode_length is equal to the number of steps
        taken and no off-by-one errors.
        Prints the execution time at the end of the test for performance check.
        """
        num_steps = 1000000
        env = AnkiVectorEnv()
        start = time.time()
        all_step_times = []
        env.reset()
        for step_num in range(num_steps):
            start_of_step = time.time()
            action = 0
            state, reward, done, _ = env.step(action)

            time_for_step = time.time() - start_of_step
            print('Step: {}. Time taken for step: {:.3f}'.
                  format(step_num, time_for_step))
            all_step_times.append(time_for_step)

            if done:
                break

        print('Time taken altogether: {}\nAverage time taken per step: {:.3f}'.format(
            time.time() - start, sum(all_step_times) / len(all_step_times)))

        self.assertTrue(len(all_step_times) == num_steps)
