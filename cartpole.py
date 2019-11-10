import math, gym, copy

env = gym.make('CartPole-v0')

# Define general class that initializes states and actions and then use it as superclass for classes below

class StateValue:
    def __init__(self, states):
        self.values = {state:0 for state in states}

    def update(self, state, new_value):
        self.values[state] = new_value

    def copy(self):
        return copy.deepcopy(self)

class StateActionObs:
    def __init__(self, states, actions):
        # Initially, each state, action pair has a value of None because we haven't sampled the environment yet
        self.immediate_obs = {(state, action):None for state in states for action in actions}

    def get_obs(self, state, action):
        return self.immediate_obs[(state, action)]


class Policy:
    def __init__(self, states, actions):
        # Initializes a uniformly random policy
        self.policy = {state:{action:1/len(actions) for action in actions} for state in states}

    def get_prob(self, action, state):
        return self.policy[state][action]


states, actions = [], env.action_space
state_value = StateValue(states, actions)
state_action_obs = StateActionObs()
policy = Policy()
gamma = .5

# POLICY EVALUATION
for _ in range(max_iterations):
    state_value_next = StateValue(states, actions)
    for state in states:
        temp = 0
        for action in env.action_space:
            reward, next_state = state_action_obs.get_obs(state, action)
            temp += policy.get_prob(action, state) * (gamma*(reward + state_value[next_state])
        state_value_next.update(state, temp)
    state_value = state_value_next.copy()





# Create buckets for the continuous values of state (x, xdot, theta, thetadot)
x_range = (-env.x_threshold, env.x_threshold)
theta_range = ()
## Use range (eg. min_theta and max_theta), if we're having 3 buckets, then split this 3-fold.
## Create a function which given observation values, places them into the correct buckets and returns the observation's

for episode in range(1,31):
    observation = env.reset()
    for timestep in range(1,101):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode {} finished after {} timesteps".format(episode,timestep))
            break
print(action)
env.close()

# Create Q(s,a) table

# How to update the q-value table
## Start q-value table with 0 for every (s,a) pair
## While starting at a certain state,
    ## While playing, at every timeframe, you are in a certain state, take an action, see the observation(i.e.next state) and --
    ## -- get the max action value for that state from current q-value table

    # To pick action, check the q(s,a'|current_state s) pair with the maximum value, and take that action

    ## Then, update the q(s,a) value in the table using the value-iteration formula
