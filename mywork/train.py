import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Model
import BotEvironment
import random
#
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# 定义Agent类
class Agent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy_net = Model.DQN(state_size, action_size)
        self.target_net = Model.DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
                return np.argmax(q_values.numpy())

    def train(self, batch):
        states, actions, rewards, next_states, done = batch
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.BoolTensor(done)

        q_values = self.policy_net(states)
        # q_values = self.policy_net(states)
        # print(q_values.shape,q_values.size())

        # next_q_values = self.target_net(next_states).max(1)[0]
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(0)[0]

        #这里要设置一下最后一步怎么办
        target_q_values = rewards + (1 - done.float()) * self.gamma * next_q_values

        loss = self.loss_fn(q_values[actions], target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练Agent
def train_agent(agent, environment, num_episodes=10000):
    #epoch
    for episode in range(num_episodes):
        state = environment.reset()
        done = False
        total_reward = 0
        loses = 0
        #惩罚过多就直接进入下一个循环
        while not done:
            actions = agent.choose_action(state+[environment.step_count]+environment.bot_state[-1])
            next_state, reward = environment.step(actions)
            total_reward += reward
            lose = agent.train((state+[environment.step_count-1]+environment.bot_state[-2], [actions], [reward], next_state+[environment.step_count]+environment.bot_state[-1], [environment.check_stop()]))
            loses += lose
            state = next_state
            if environment.check_stop():
                print(environment.bot_state)
                done = True
        agent.update_target_net()
        print("Episode:", episode, "Total Reward:", total_reward, "loses:", lose)


if __name__ == '__main__':
    env = BotEvironment.Environment()
    agent = Agent(state_size=8, action_size=25)

    # 训练Agent
    train_agent(agent, env)
