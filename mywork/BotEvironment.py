action2positioon = {
    0: [0, 0],
    1: [0, 1],
    2: [0, 2],
    3: [0, 3],
    4: [0, 4],
    5: [1, 0],
    6: [1, 1],
    7: [1, 2],
    8: [1, 3],
    9: [1, 4],
    10: [2, 0],
    11: [2, 1],
    12: [2, 2],
    13: [2, 3],
    14: [2, 4],
    15: [3, 0],
    16: [3, 1],
    17: [3, 2],
    18: [3, 3],
    19: [3, 4],
    20: [4, 0],
    21: [4, 1],
    22: [4, 2],
    23: [4, 3],
    24: [4, 4],
}


class Environment:
    def __init__(self, num_steps=15):
        self.num_steps = num_steps
        self.step_count = 1
        self.state = [0, 0, 0, 0, 1]  #第一步的格子情况
        self.bot_state = [[0, 3]]  # 初始状态：两个机器人的位置
        self.obstacles_1 = {1: [4], 2: [1], 3: [2], 4: [4], 5: [0, 3], 6: [1], 7: [2], 8: [3], 9: [0, 2],
                            10: [4], 11: [0, 2], 12: [1], 13: [2], 14: [4], 15: [0, 2]}  # 障碍物方框
        self.obstacles_2 = {2: [4], 3: [1], 6: [4], 7: [1], 12: [4], 13: [1]}

    def reset(self):
        self.bot_state = [[0, 3]]
        self.step_count = 1
        self.state = [0, 0, 0, 0, 1]
        return self.state

    def check_stop(self):
        if self.step_count >= self.num_steps:
            return True
        else:
            return False

    def list_render(self, col):
        block_list = [0 for _ in range(5)]
        if col in self.obstacles_1:
            obstacles_list = self.obstacles_1[col]
            for i in obstacles_list:
                block_list[i] = 1

        if col in self.obstacles_2:
            obstacles_list = self.obstacles_2[col]
            for i in obstacles_list:
                block_list[i] = 2
        return block_list

    def step(self, actions):
        self.step_count += 1
        rewards = 0
        block_list = self.list_render(self.step_count)

        #action是一个数，position做具体的转换
        next_position = action2positioon[actions]

        ckeck_obstacles = False

        if next_position[0] == next_position[1]:
            rewards -= 10
            ckeck_obstacles = True
        if block_list[next_position[0]] == 1:
            rewards -= 10
            ckeck_obstacles = True
        if block_list[next_position[1]] == 1:
            rewards -= 10
            ckeck_obstacles = True
        if block_list[next_position[0]] == 2:
            rewards -= 4
            ckeck_obstacles = True
        if block_list[next_position[1]] == 2:
            rewards -= 4
            ckeck_obstacles = True

        if not ckeck_obstacles:
            rewards += 8

        self.bot_state.append(next_position)
        next_state = self.list_render(self.step_count + 1)

        return next_state, rewards
