import math, gym, copy
import operator
import random
import pickle
from cartpole_env import *

cartpole = CartPoleEnv()

def approximate(raw_state):
    array = list(raw_state)
    return tuple([round(num, 2) + 0 for num in array])

def generate(lower_bound, upper_bound, interval):
    output = [lower_bound]
    while lower_bound <= upper_bound:
        lower_bound += interval
        output.append( float( round(lower_bound,1) + 0 ) )
    return output

def generate_states(x_approx, x_dot_approx, theta_approx, theta_dot_approx):
    states = []
    for x in x_approx:
        for x_dot in x_dot_approx:
            for theta in theta_approx:
                for theta_dot in theta_dot_approx:
                    states.append((x, x_dot, theta, theta_dot))
    return states

def get_value(state, value_function):
    try:
        next_state_value = value_function[state]
    except KeyError: # if next_state is not in value_function, assume it's a 'dead' state.
        next_state_value = -500
    return next_state_value

def create_transition_reward_function(states, actions, env):
    table = {}
    for state in states:
        for action in actions:
            env.reset(state)
            obs, reward, done, info = env.step(action)
            table[(state, action)] = {'reward':reward, 'next_state':approximate(obs)}
    return table

def evaluate_policy(state, actions, transition_and_reward_function, policy, value_function, gamma=1.0):
    new_val = 0
    for action in actions:
        reward, next_state = transition_and_reward_function[(state, action)].values()
        next_state_value = get_value(next_state, value_function)
        new_val += policy[state][action] * (reward + gamma*next_state_value)
    return new_val

def improve_policy(states, actions, transition_and_reward_function, value_function, gamma=1.0):
    new_policy = {}
    for state in states:
        action_values = {}
        for action in actions:
            reward, next_state = transition_and_reward_function[(state, action)].values()
            action_values[action] = reward + gamma*get_value(next_state, value_function)
        greedy_action, value = max(action_values.items(), key= lambda pair: pair[1])
        new_policy[state] = {action:1 if action is greedy_action else 0 for action in actions}
    return new_policy


# Policy iteration
def policy_iteration(states, actions, transition_and_reward_function, policy, value_function, number_of_iterations = 50):
    new_value_function = {}
    for _ in range(number_of_iterations):
        print("iteration_number:",_)
        # Evaluate every state under current policy
        for state in states:
            new_value_function[state] = evaluate_policy(state, actions, transition_and_reward_function, policy, value_function)
        # Policy improvement
        policy = improve_policy(states, actions, transition_and_reward_function, new_value_function)
        value_function = new_value_function
    return policy

def get_optimal_action(state, optimal_policy):
    try:
        greedy_action, prob = max(optimal_policy[state].items(), key= lambda pair: pair[1])
        return greedy_action
    except KeyError:
        return random.randint(0,1)



if __name__ == '__main__':
    # Setting up
    x_approx = generate(-0.5, 0.5, 0.01)
    x_dot_approx = generate(-2.0, 2.0, 0.1)
    theta_approx = generate(-0.1, 0.1, 0.01) # theta_thres is 24deg, i.e. 24*2*pi/360 radians
    theta_dot_approx = generate(-3.0, 3.0, 0.1)
    states = generate_states(x_approx, x_dot_approx, theta_approx, theta_dot_approx)
    actions = [0,1]
    transition_and_reward_function = create_transition_reward_function(states, actions, cartpole)
    starting_policy = {state:{0:0.5, 1:0.5} for state in states}
    value_function = {state:0 for state in states}
    optimal_policy = policy_iteration(states, actions, transition_and_reward_function, starting_policy, value_function)

    # Saving the optimal_policy on a file
    optimal = open('optimal.pkl','wb')
    pickle.dump(optimal_policy, optimal)
    optimal.close()
    print("done finding optimal_policy")

    # Running test episodes with optimal_policy
    total_steps = 0
    num_episodes = 10
    for episode in range(0,num_episodes):
        observation = cartpole.reset()
        for timestep in range(1,101):
            cartpole.render()
            action = get_optimal_action(approximate(observation), optimal_policy)
            observation, reward, done, info = cartpole.step(action)
            print("Observation:", observation, ' ', "Action:", action)
            if done:
                print("Episode {} finished after {} timesteps".format(episode,timestep))
                print(observation)
                total_steps += timestep
                break

    print("average time upright per episode:", total_steps/num_episodes)
    cartpole.close()
