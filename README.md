
# Cartpole-balancing

The goal of this project was to apply policy iteration to the problem of balancing a pole on a moving cart. The goal is simple: as a controller you can only control the cart at every timestep by either pushing it left or right. There's a limit to how far the cart can move in both the left and right direction from its starting position (center). There's also a limit to the angle that the pole can fall from its vertical axis. More comprehensive problem definition can be found [here](https://github.com/openai/gym/wiki/CartPole-v0)

![cartpole](/assets/cartpole.png)

## Policy iteration
Policy iteration is a framework for nearly all control methods used in reinforcement learning. It consists of 2 processes: policy evaluation and policy improvement. These two procedures feedback into each other resulting into a new improved policy that always performs equally or better than the old one on the given environment.

## Challenge and solution
The states of this environment are represented using 4 continuous variables: cart position, cart velocity, pole angle, and pole velocity at tip. A common practice is to discretize the state space. Depending on how granular this process is, the results can be quite different. My approach was two-fold.
1. Ignore all the dead states i.e. ones in which the cart position and/or pole angle are beyond their specified thresholds.
2. Round cart-position and pole-angle to .01 and the remaining variables to .1

With these, I use bellman equation for evaluating each state and then do greedy policy improvement.

# Environment
My implementation uses a slightly modified version of OpenAI's gym cart-pole environment that can be found as cartpole_env.py. The modification is the addition of an optional argument to the reset method. With this, the environment can be reset to any starting state. I do this in order to build a transition table mapping all state and action pairs to their respective rewards and the next observed states.

### Installing gym
```
pip3 install gym
```
For more instructions: [gym](https://gym.openai.com/docs/)


## Running the tests

After running cartpole.py, the resulting policy is stored in a file with .pkl format. This policy can then be run on the environment without the whole policy iteration process starting afresh.


## Acknowledgments

*  Sutton, R. & Barto, A. _Reinforcement Learning: an Introduction_ (MIT Press, 2018)
* Prof. David Silver's RL lecture(s): [here](https://youtu.be/Nd1-UUMVfz4)
