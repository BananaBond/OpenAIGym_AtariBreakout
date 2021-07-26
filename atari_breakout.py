#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np
import gym

class AtariBreakoutEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env = gym.make("Breakout-v0")
        directory = "recordings"
        self.env = gym.wrappers.Monitor(self.env, directory, video_callable=lambda episode_id: episode_id%5==0, force=True)
        self.env.seed(0)

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        reward = 0.0
        #Called at the start of every gym env to reset to the initial observation stage
        #returns observation that is specific to environment youre using
        observation = self.env.reset()
        is_terminal = False
        observation = observation.astype('float32')
        # observation = observation[:,:,1]
        self.reward_obs_term = (reward, observation, is_terminal)

        # return first state observation from the environment
        return self.reward_obs_term[1]



    def env_step(self, action, renderEnv):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        action = int(action)

        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)
        current_state = current_state.astype('float32')
        # sending only the green component of the RGB image as a vector
        # current_state = current_state[:,:,1]

        self.reward_obs_term = (reward, current_state, is_terminal)
        if renderEnv:
            self.env.render()

        return self.reward_obs_term
