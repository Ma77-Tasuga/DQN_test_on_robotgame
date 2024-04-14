import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
# 游戏参数
ROWS = 5
COLS = 14
NUM_ACTIONS = 5
START_POS = (0, 0)
START_POS_2 = (3, 0)
OBSTACLE_1_POS = [(4,0),(1, 1), (2, 2), (4, 3),(0,4),(3,4),(1,5),(2,6),(3,7),(0,8),(2,8),(4,9),(0,10),(2,10),(1,11),(2,12),(4,13),(0,14),(2,14)]
OBSTACLE_2_POS = [(4,1),(1,2),(5,5),(1,6), (4, 11), (1, 12)]
REWARD_EMPTY = 8 #8
PENALTY_COLLISION_1 = -10  #-10
PENALTY_COLLISION_2 = -4  #-4
COLLISION_PENALTY=-2 #-8
EPISODES = 5000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.01

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class QNetwork_2(nn.Module):
    def __init__(self):
        super(QNetwork_2, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def choose_action(state_arg, q_net_arg, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        with torch.no_grad():
            q_values = q_net_arg(state_arg)
            return torch.argmax(q_values).item()
def calculate_reward(action_robot1, action_robot2):
    reward = 0
    reward2 = 0
    if action_robot1 == action_robot2:
        # 碰撞扣分
        reward += COLLISION_PENALTY
        reward2 += COLLISION_PENALTY
    if action_robot1 in OBSTACLE_1_POS:
        # if state in OBSTACLE_1_POS:
        reward += PENALTY_COLLISION_1
    elif action_robot1 in OBSTACLE_2_POS:
        # elif state in OBSTACLE_2_POS:
        reward += PENALTY_COLLISION_2
    else:
        reward += REWARD_EMPTY
    if action_robot2 in OBSTACLE_1_POS:
        reward2 += PENALTY_COLLISION_1
    elif action_robot2 in OBSTACLE_2_POS:
        reward2 += PENALTY_COLLISION_2
    else:
        reward2 += REWARD_EMPTY
    return reward,reward2
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

def experience_replay(replay_buffer,q_net, optimizer,criterion,batch_size ):
    batch = replay_buffer.sample_batch(batch_size)
    state, action, reward, next_state = zip(*batch)
    state = torch.tensor(torch.cat(state), dtype=torch.float32)
    action =  torch.tensor(action, dtype=torch.int64)
    reward = torch.tensor(reward, dtype=torch.float32)
    next_state =  torch.tensor(torch.cat(next_state), dtype=torch.float32)
    with torch.no_grad():
        next_q_values = q_net(next_state)
        max_next_q_value =torch.max(next_q_values,dim=-1).values
        target_q_value = torch.tensor(reward + GAMMA * max_next_q_value).float()

    predicted_q_value = q_net(state)
    predicted = torch.gather( predicted_q_value,dim=1,index=action.unsqueeze(-1))
    loss = criterion( predicted , target_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    REPLAY_BUFFER_SIZE = 100
    batch_size = 1
    # replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    # replay_buffer_2 = ReplayBuffer(REPLAY_BUFFER_SIZE)
    q_net = QNetwork()
    q_net_2 = QNetwork_2()
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    optimizer_2 = optim.Adam(q_net_2.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    total_reward_list = []
    total_reward_list_2 = []
    for episode in range(EPISODES):
        state = torch.tensor(START_POS, dtype=torch.float32).view(1, -1)
        state_2 = torch.tensor(START_POS_2, dtype=torch.float32).view(1, -1)
        total_reward = REWARD_EMPTY
        total_reward_2 = REWARD_EMPTY
        STATS=[]
        STATS.append(state)
        STATS_2 = []
        STATS_2.append(state_2)
        # reward=REWARD_EMPTY
        for step in range(0,COLS):
            action = choose_action(state, q_net, EPSILON)
            action_2 = choose_action(state_2, q_net_2, EPSILON)
            next_state = torch.tensor((action,state[0, 1] + 1), dtype=torch.float32).view(1, -1)
            next_state_2 = torch.tensor((action_2, state_2[0, 1] + 1), dtype=torch.float32).view(1, -1)

            reward, reward_2 = calculate_reward((action,state[0, 1].item() + 1),(action_2,state_2[0, 1].item() + 1))

            total_reward += reward
            total_reward_2 += reward_2
            if total_reward>=100:
                reward +=20
            if total_reward_2>=100:
                reward_2 +=20
            # replay_buffer.add_experience((state, action, reward, next_state))
            # replay_buffer_2.add_experience((state_2, action_2, reward_2, next_state_2))

            # if len(replay_buffer.buffer) >= batch_size:
            #     experience_replay(replay_buffer, q_net ,optimizer,criterion, batch_size)
                # experience_replay(replay_buffer_2, q_net_2, optimizer_2, criterion, batch_size)
            with torch.no_grad():
                next_q_values = q_net(next_state)
                next_q_values_2 = q_net_2(next_state_2)
                max_next_q_value = torch.max(next_q_values, dim=-1).values
                max_next_q_value_2 = torch.max(next_q_values_2, dim=-1).values
                target_q_value = torch.tensor(reward + GAMMA * max_next_q_value).float()
                target_q_value_2 = torch.tensor(reward_2 + GAMMA * max_next_q_value_2).float()

            predicted_q_value = q_net(state)
            predicted_q_value_2 = q_net_2(state_2)
            predicted = predicted_q_value[0][action]
            predicted_2 = predicted_q_value_2[0][action_2]
            #
            loss = criterion(predicted, target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_2 = criterion(predicted_2, target_q_value_2)
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            state = next_state
            state_2 = next_state_2
            STATS.append(next_state)
            STATS_2.append(next_state_2)
        total_reward_list.append(total_reward)
        total_reward_list_2.append(total_reward_2)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        print(f"Episode {episode + 1}, Total Reward_2: {total_reward_2}")

    x = list(range(1, len(total_reward_list) + 1))
    plt.plot(x, total_reward_list, marker='o')
    plt.plot(x, total_reward_list_2, marker='s')
    # 添加标题和标签
    plt.title('折线图示例')
    plt.xlabel('X 轴')
    plt.ylabel('Y 轴')
    # 显示图例
    plt.legend(['数据'])
    # 显示图形
    plt.show()
    print(STATS)
    print(STATS_2)
if __name__ == "__main__":
    main()
