import sys
import numpy as np
import random
import gym

# constants
ACTION_SPACE = [0,1]


# Change this so that it can take in arguments in any format, strip them down, turn them to numpy arrays and merge them
def get_feature_vector(state, action, order=2):
    """
    Inputs:
        - state: numpy array of state variables
        - action: numpy array of an action
        - order: desired highest degree in feature vector
    """

    # TODO: Be able to calculate feature vectors of various degrees

    return np.append(state, action)

def e_greedy_action_selection(action_value_estimator, state, num_features, epsilon=0.1):
    Q = {}
    for action in ACTION_SPACE:
        feature_vector = get_feature_vector(state, action)
        action_value = action_value_estimator.get_value(feature_vector)
        Q[tuple(feature_vector)] = action_value
    Q_sorted = sorted(Q.items(), key = lambda pair:pair[1], reverse=True)  # descending order i.e. largest --> smallest
    greedy_action, greedy_value = Q_sorted[0]
    distribution = [1 - epsilon if val is greedy_value else epsilon/(len(ACTION_SPACE)-1) for feature_vector,val in Q_sorted]
    feature_vectors = [feature_vector for feature_vector,val in Q_sorted]
    f = random.choices(feature_vectors,distribution)[0]  # feature_vector selected epsilon-greedily
    return int(f[-1]) # get last item in f which is the associated action

class Estimator:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = self.initialize_weights()

    def initialize_weights(self, zeros=True):
        if zeros:
            return np.zeros(self.num_features)
        return np.random.rand(self.num_features)

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

    def learn(self, env, training_episodes, step_size, gamma):
        for episode in range(1,training_episodes):
            current_state = env.reset()
            current_action = e_greedy_action_selection(self, current_state, self.num_features)
            done = False
            while not done:
                next_state, reward, done, info = env.step(current_action)
                next_action = e_greedy_action_selection(self, next_state, self.num_features)
                self.semi_sgd_weight_update(get_feature_vector(current_state, np.array([current_action])), reward, get_feature_vector(next_state, np.array([next_action])), step_size, gamma)
                current_state = next_state
                current_action = next_action
            print("Training episode:{}/{}".format(episode, training_episodes))
        return

def pick_action(value_function, observation):
    action_to_value = {}
    for action in ACTION_SPACE:
        action_to_value[action] = value_function.get_value(get_feature_vector(observation, np.array([action])))
    action = max(action_to_value.items(), key = lambda pair:pair[1])[0]
    return action

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    value_function = Estimator(5)
    training_episodes, step_size, gamma = 10000, 0.01, 0.999
    value_function.learn(env, training_episodes, step_size, gamma)
    test_trials = 100
    for episode in range(test_trials):
        observation = env.reset()
        for t in range(100):
            env.render()
            ## CHANGE THISS LATTERR
            action = pick_action(value_function, observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
