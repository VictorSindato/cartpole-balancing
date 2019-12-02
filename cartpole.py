import math, gym, copy
import operator
from cartpole_env import *

cartpole = CartPoleEnv()

def approximate(raw_state):
    array = list(raw_state)
    return tuple([round(array[i], 1 - i%2) + 0 for i in range(len(array))])
    # return tuple([round(component, 1) + 0 for component in array])

def generate(lower_bound, upper_bound, interval):
    output = []
    while lower_bound <= upper_bound:
        output.append( float( round(lower_bound,1) + 0 ) )
        lower_bound += interval
    return output

def generate_states(x_approx, x_dot_approx, theta_approx, theta_dot_approx):
    states = []
    for x in x_approx:
        for x_dot in x_dot_approx:
            for theta in theta_approx:
                for theta_dot in theta_dot_approx:
                    states.append((x, x_dot, theta, theta_dot))
    return states

# Just have env as an argument and deduce actions from env
def create_transition_reward_function(states, actions, env):
    function = {}
    for state in states:
        for action in actions:
            env.reset(state)
            obs, reward, done, info = env.step(action)
            function[(state, action)] = {'reward':reward, 'next_state':approximate(obs)}
    return function

def policy_eval(state, actions, transition_and_reward_function, policy, value_function, gamma=1.0):
    new_val = 0
    for action in actions:
        reward = transition_and_reward_function[(state, action)]['reward']
        next_state = transition_and_reward_function[(state, action)]['next_state']
        try:
            next_state_value = value_function[next_state]
        except KeyError:
            next_state_value = 0
        new_val += policy[state][action] * (reward + gamma*next_state_value)
    return new_val

def greedy_policy(states, actions, transition_and_reward_function, value_function):
    """
    Input: states, actions, transition_and_reward_function, value_function
    Output: greedy_policy
    """
    new_policy, optimal_actions = {}, {}
    for state in states:
        action_to_next_state_value = {}
        new_policy[state] = {}
        for action in actions:
            next_state = transition_and_reward_function[(state, action)]['next_state']
            try:
                next_state_value = value_function[next_state]
            except KeyError:
                next_state_value = 0
            action_to_next_state_value[action] = next_state_value
        # if there are two max values...just pick one (assumption that needs to be fixed)

        greedy_action = max(action_to_next_state_value.items(), key= lambda pair: pair[1])
        for action in actions:
            if action == greedy_action:
                new_policy[state][action] = 1
                optimal_actions[state] = action
            else:
                new_policy[state][action] = 0
    return (new_policy, optimal_actions)


# Policy iteration
def policy_iteration(states, actions, transition_and_reward_function, policy, value_function, number_of_iterations = 10):
    new_value_function = {}
    for _ in range(number_of_iterations):

        print("iteration_number:",_)

        # Policy evaluation
        for state in states:
            new_value_function[state] = policy_eval(state, actions, transition_and_reward_function, policy, value_function)
        # Policy improvement
        policy, optimal_actions = greedy_policy(states, actions, transition_and_reward_function, new_value_function)
        value_function = new_value_function

    return (policy, optimal_actions)

x_approx = generate(-4.8, 4.8, 0.1)
x_dot_approx = generate(-5, 5, 1)
theta_approx = generate(-0.5, 0.5, 0.1) # theta_thres is 24deg, i.e. 24*2*pi/360 radians
theta_dot_approx = generate(-5, 5, 1)
states = generate_states(x_approx, x_dot_approx, theta_approx, theta_dot_approx)
transition_and_reward_function = create_transition_reward_function(states, [0,1], cartpole)
starting_policy = {state:{0:0.5, 1:0.5} for state in states}
value_function = {state:0 for state in states}
actions = [0,1]
optimal_policy, optimal_actions = policy_iteration(states, actions, transition_and_reward_function, starting_policy, value_function, 10)

# for a,b in optimal_actions.items():
#     print("state:", a, "optimal_action:",b)

total_steps = 0
num_episodes = 10

for episode in range(0,num_episodes):
    observation = cartpole.reset()
    for timestep in range(1,101):
        cartpole.render()
        action = optimal_actions[approximate(observation)]
        print("Observation:", observation, ' ', "Action:", action)
        observation, reward, done, info = cartpole.step(action)
        if done:
            print("Episode {} finished after {} timesteps".format(episode,timestep))
            print(observation)
            total_steps += timestep
            break

print("average seconds per episode:", total_steps/num_episodes)
cartpole.close()

# for a,b in transition_and_reward_function.items():
#     print(a,b)


# benchmark
# total_steps = 0
# for episode in range(0,num_episodes):
#     observation = cartpole.reset()
#     for timestep in range(1,101):
#         cartpole.render()
#         if tuple(observation)[2] < 0:
#             action = 0
#         else:
#             action = 1
#         observation, reward, done, info = cartpole.step(action)
#         if done:
#             print("Episode {} finished after {} timesteps".format(episode,timestep))
#             print(observation)
#             total_steps += timestep
#             break
# print("Average time upright per episode:", total_steps/num_episodes)
# cartpole.close()
