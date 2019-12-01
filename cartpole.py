import math, gym, copy
import operator

cartpole = gym.make('CartPole-v0')


def approximate(raw_state):
    """
    Input: numpy array in format array([x, x_dot, theta, theta_dot])
    Output: tuple of input's values rounded
    """
    # return (round(component, 1) for component in list(raw_state))
    return tuple([float(round(list(raw_state)[i], 1-i%2)+0) for i in range(len(list(raw_state)))])


def generate(lower_bound, upper_bound, interval):
    output = []
    while lower_bound <= upper_bound:
        output.append( float( round(lower_bound,1) + 0 ) )
        lower_bound += interval
    return output

x_approx = generate(-4.8, 4.8, 0.1)
x_dot_approx = generate(-10, 10, 0.5)
theta_approx = generate(-0.5, 0.5, 0.1) # theta_thres is 24deg, i.e. 24*2*pi/360 radians
theta_dot_approx = generate(-5, 5, 0.5)
states = [(x, x_dot, theta, theta_dot) for x in x_approx for x_dot in x_dot_approx for theta in theta_approx for theta_dot in theta_dot_approx]

# Just have env as an argument and deduce actions from env
def create_transition_reward_function(states, actions, env):
    transition_and_reward_function = {}
    env.reset()
    for state in states:
        for action in actions:
            env.state = state
            obs, reward, done, info = env.step(action)
            env.reset()
            transition_and_reward_function[(state, action)] = {'reward':reward, 'next_state':tuple(approximate(obs))}
    return transition_and_reward_function

transition_and_reward_function = create_transition_reward_function(states, [0,1], cartpole)

def policy_eval(policy, value_function, state, actions, transition_and_reward_function, gamma=1.0):
    new_val = 0
    for action in actions:
        reward = transition_and_reward_function[(state, action)]['reward']
        next_state = transition_and_reward_function[(state, action)]['next_state']
        new_val += policy[state][action] * (reward + gamma*value_function[next_state])
    return new_val

def greedy_policy(states, actions, transition_and_reward_function, value_function, current_policy):
    """
    Input: states, actions, transition_and_reward_function, value_function, current_policy
    Output: greedy_policy
    """
    new_policy = {}
    for state in states:
        action_to_next_state_value = {}
        for action in actions:
            next_state = transition_and_reward_function[(state, action)]['next_state']
            next_state_value = new_value_function[next_state]
            action_to_next_state_value[action] = next_state_value
        # if there are two max values...just pick one (assumption that needs to be fixed)
        greedy_action = max(action_to_next_state_value.items(), key=operator.itemgetter(1))[0]
        for action in actions:
            if action == greedy_action:
                new_policy[state][action] = 1
                optimal_actions[state] = action
            else:
                new_policy[state][action] = 0

    return new_policy


# Policy iteration
def policy_iteration(states, actions, transition_and_reward_function, policy, value_function, number_of_iterations = 10):
    optimal_actions = {}
    old_value_function = value_function
    new_value_function = {}
    new_policy = policy
    for _ in range(number_of_iterations):
        print("iteration_number:",_)
        # Policy evaluation
        for state in states:
            new_value_function[state] = policy_eval(new_policy, old_value_function, state, actions, transition_and_reward_function)

        # Policy improvement
        new_policy = greedy_policy(states, actions, transition_and_reward_function, new_value_function, new_policy)
        old_value_function = new_value_function

    return (new_policy, optimal_actions)

starting_policy = {state:{0:0.5, 1:0.5} for state in states}
value_function = {state:0 for state in states}
actions = [0,1]


optimal_policy, optimal_actions = policy_iteration(states, actions, transition_and_reward_function, starting_policy, value_function, 100)
print('----------------------------')
#
print('-----------------------------')

total_steps = 0
num_episodes = 10

for episode in range(1,num_episodes+1):
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


# total_steps = 0
#
# # HANDMADE
# for episode in range(1,31):
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
# print("Total seconds pole was upright in 30 episodes:", total_steps)
# cartpole.close()
