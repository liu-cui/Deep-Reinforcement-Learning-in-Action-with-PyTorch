"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
"""

from common.log import logging
import random
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt


matplotlib.style.use("ggplot")
np.random.seed(2)  # reproducible

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ["left", "right"]  # available actions
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 50  # maximum episodes
FRESH_TIME = 0.3   # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q table init
        columns=actions   # actions name
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or (state_actions == 0).all():  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    if A == "right":
        if S == N_STATES - 2:  # terminate
            S_ = "terminal"
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]
    if S == "terminal":
        interation = "Episode %s: total steps = %s" % (episode + 1, step_counter)
        time.sleep(0.5)
    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        time.sleep(FRESH_TIME)

def run_hello_agent():
    q_table = build_q_table(N_STATES, ACTIONS)
    episodes = []
    rewards = []
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        total_reward = 0

        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action and get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != "terminal":
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state
            update_env(S, episode, step_counter + 1)
            step_counter += 1
            total_reward += q_target
        episodes.append(episode+1)
        rewards.append(total_reward)
        logging.info(f"episode = {episode + 1}, steps = {step_counter}, reward = {total_reward}")
    plt.plot(episodes, rewards)
    # plt.show()
    plt.savefig("../agents/images/treasure.png")
    return q_table


if __name__ == '__main__':
    logging.info(f"hello world, start train your agent!")
    run_hello_agent()



