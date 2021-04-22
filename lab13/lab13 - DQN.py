import importlib
from collections import defaultdict
import torch
import numpy
from myDQN import DQN, Memory

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
    buffer = 3
    memory = Memory(buffer)
    model = DQN(number_action=n_action)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    criterion = torch.nn.MSELoss()
    model.to(device)
    model.train()

    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        ep_loss = 0
        if episode % 10000 == 9999:
            print(f"{episode} has loss {ep_loss}")
            # print("episode: ", episode + 1)
        state = env.reset()
        memory.reset()
        state_o = state
        state = hash(tuple(state.reshape(-1)))
        is_done = False
        with torch.set_grad_enabled(True):
            while not is_done:
                optimizer.zero_grad()
                if env.to_play() == player:
                    available_action = env.legal_actions()
                    action = epsilon_greedy_policy(state, Q, available_action)
                    next_state, reward, is_done = env.step(action)
                    next_state_o = next_state
                    next_state = hash(tuple(next_state.reshape(-1)))
                    td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
                    Q[state][action] += alpha * td_delta
                else:
                    action = env.expert_agent()
                    next_state, reward, is_done = env.step(action)
                    next_state_o = next_state
                    next_state = hash(tuple(next_state.reshape(-1)))

                    if is_done:
                        reward = -reward
                        td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
                        Q[state][action] += alpha * td_delta

                length_episode[episode] += 1
                total_reward_episode[episode] += reward
                memory.add_memory(state_o, action, reward, next_state_o)
                
                e = memory.get_memory_random()
                y = torch.as_tensor(e[2]).float()
                if is_done == False:
                    data = torch.as_tensor(e[3].reshape(-1).astype(float)).float()
                    data.requires_grad = True
                    actions = torch.nn.functional.softmax( model(data.to(device)), dim=0)
                    # print("outputs:", outputs.shape)
                    # print("output:", torch.argmax(outputs) )
                    y = e[2] + gamma * Q[hash(tuple(e[3].reshape(-1)))][torch.argmax(actions)]
                # print("y:" , y)

                # a = torch.argmax(torch.nn.functional.softmax(model(e[0]),dim=0))
                # print(a.view(1,-1),"=|||||=",y)
                y.requires_grad = True
                # print(y.requires_grad)
                loss = criterion(Q[hash(tuple(e[0].reshape(-1)))][e[1]], y.view(1,-1))
                ep_loss += loss
                # loss.requres_grad = True
                loss.backward()
                optimizer.step()
                if(is_done):
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

