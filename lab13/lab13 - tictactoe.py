# import importlib

# game_name = 'tictactoe'
# game_module = importlib.import_module("games." + game_name)
# env = game_module.Game()

# env.reset()

# done = False

# while done == False:
#     action = env.human_to_action()
#     if(action == 'q'): break
#     next_state, reward, done = env.step(action)
#     env.render()

# print("The Winner is: ", env.player)
# # env.have_winner()


import importlib
from collections import defaultdict
import torch
import numpy

game_name = 'tictactoe'
game_module = importlib.import_module("games." + game_name)
env = game_module.Game()

env.reset()

# defining epsilon-greedy policy
def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q, available_actions):
        probs = torch.ones(n_action) * epsilon / n_action
        # print(probs)
        # print(state)
        # print(Q[state])
        best_action = torch.argmax(Q[state]).item()
        if not(best_action in available_actions):
            best_action = -1
            Q_max = -800000000
            for i in range(n_action):
                if i in available_actions and Q_max < Q[state][i]:
                    Q_max = Q[state][i]
                    best_action = i
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, gamma, n_episode, alpha, player):
    """
    Obtain the optimal policy with off-policy Q-learning method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = 9
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        if episode % 10000 == 9999:
            print("episode: ", episode + 1)
        state = env.reset()
        state = hash(tuple(state.reshape(-1)))

        is_done = False
        while not is_done:
            if env.to_play() == player:
                available_action = env.legal_actions()
                action = epsilon_greedy_policy(state, Q, available_action)
                next_state, reward, is_done = env.step(action)
                next_state = hash(tuple(next_state.reshape(-1)))
                td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
                Q[state][action] += alpha * td_delta
            else:
                action = env.expert_agent()
                next_state, reward, is_done = env.step(action)
                next_state = hash(tuple(next_state.reshape(-1)))

                if is_done:
                    reward = -reward
                    td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
                    Q[state][action] += alpha * td_delta

            length_episode[episode] += 1
            total_reward_episode[episode] += reward

            if is_done:
                break
            state = next_state

    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

gamma = 1

n_episode = 1000000
# n_episode = 10
alpha = 0.4

epsilon = 0.1

available_action = env.legal_actions()
epsilon_greedy_policy = gen_epsilon_greedy_policy(9, epsilon)

length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

# agent play first
optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha, 1)

torch.save(optimal_policy,'./q-learning/opt_policy.pth' )

torch.save(dict(optimal_Q),'./q-learning/opt_Q.pth' )
print('The optimal policy:\n', optimal_policy)
print('The optimal Q:\n', optimal_Q)

