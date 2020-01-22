import sys
import numpy as np
import random
import gym


def get_feature_vector(state, action, order=2):
    """
    Inputs:
        - state: numpy array of state variables
        - action: numpy array of an action
        - order: desired highest degree in feature vector
    """

    # TODO: Be able to calculate feature vectors of various degrees

    return np.append(state, action)


class Action_Value_Estimator:
    def __init__(self, num_state_variables):
        self.weights = initialize_weights()

    def initialize_weights(self, zeros=True):
        if zeros:
            return np.zeros(num_state_variables)
        return np.random.rand(num_state_variables)

    def get_value(self, feature_vector):
        """
        Input: [numpy array] feature_vector --> eg. np.array([1,2,3,4])
        """
        return np.dot(feature_vector, self.weights)

    # Sarsa
    def semi_sgd_weight_update(self, current_feature_vector, reward, next_feature_vector, step_size, gamma):
        """
        Inputs:
            - [numpy array] current_feature_vector, next_feature_vector --> eg. np.array([1,2,3,4])
            - [float] reward, step_size, gamma
        """
        target_error = reward + gamma*self.get_value(next_feature_vector) - self.get_value(current_feature_vector)
        self.weights += step_size * target_error * current_feature_vector   # gradient of value of current feature vector with respect to weights is the feature vector itself
        return


def e_greedy_action_selection(value_estimator, state, action, epsilon=0.1):
    return


def learn(training_episodes,):
    for episode in range(training_episodes):
        current_state, current_action = env.reset(), e_greedy_action_selection()
        done = False
        while not done:
            next_state, reward, done, info = env.step(current_action)
            semi_sgd_weight_update()
            current_state = next_state
            current_action = next_action

        semi_sgd_weight_update()



if __name__ == '__main__':
    states, actions = all_states(maze_width, maze_height), all_actions()
    policy = using_sarsa(env, states, actions)
    test_trials = 200
    for episode in range(test_trials):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = policy[tuple(observation)]
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
