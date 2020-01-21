import sys
import numpy as np
import random
import gym


class Value_Estimator:
    def __init__(self, num_state_variables):
        self.weights = initialize_weights()

    def initialize_weights(zeros=True):
        if zeros:
            return np.zeros(num_state_variables)
        return np.random.rand(num_state_variables)

    def get_value(self, state, action):
        return



def semi_sgd_weight_update(state, weights, step_size, target_error):
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
