from enum import Enum
import numpy as np


class State:
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    def __init__(self, grid, move_prob=0.8):
        """
        智能体环境
        :param grid: 二维数组，元素值表示对应格子的属性：0=普通格子，-1=有危险的格子（游戏结束），1=有奖励的格子（游戏结束），9=被屏蔽的格子
        :param move_prob:
        """
        self.grid = grid
        self.agent_state = State()

        # 默认奖励是负数，施加初始位置惩罚，即意味着智能体必须快速到达终点
        self.default_reward = -0.04

        # 智能体能够以 move_prob 的概率向所选方向移动
        # 如果概率值在(1 - move_prob)内
        # 则意味着智能体将移动到不同的方向
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        """
        状态空间
        :return: 状态集合
        """
        states = []
        for row in range(self.row_length):
            for col in range(self.column_length):
                # state中不包括被屏蔽的格子
                if self.grid[row][col] != 9:
                    states.append(State(row, col))
        return states

    def transit_func(self, state, action):
        """
        状态转移函数
        :param state: 智能体当前所处状态
        :param action: 智能体在当前状态下施发的动作
        :return: 每个动作的执行概率
        """
        transition_probs = {}
        if not self.can_action_at(state):
            # 已经到达游戏结束的格子
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """
        是否能够施发动作的状态
        :param state: 当前所处的状态
        :return: Ture/False
        """
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can not move from here!")

        next_state = state.clone()

        # 执行动作（移动）
        if action == Action.UP:
            next_state.row -= 1
        elif action == action.DOWN:
            next_state.row += 1
        elif action == action.LEFT:
            next_state.column -= 1
        elif action == action.RIGHT:
            next_state.column += 1

        # 检查下一个状态是否在grid之外
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # 检查智能体是否到了被屏蔽的格子
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """奖励函数"""
        reward = self.default_reward
        done = False

        # 检查下一种状态的属性
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # 获取奖励，游戏结束
            reward = 1
            done = True
        elif attribute == -1:
            # 遇到危险，游戏结束
            reward = -1
            done = True

        return reward, done

    def reset(self):
        """初始化智能体的位置（使之回到左下角）"""
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, None

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
