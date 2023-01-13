'''
Author: 孙凯威
Date: 2021-11-11 15:08:34
LastEditTime: 2022-02-12 10:11:21
LastEditors: Please set LastEditors
Description: 单个点分别作为邻域的圆心构建环境
FilePath: \ship_design\env_design\env1.py
'''
import gym
from gym import spaces
import numpy as np
import json
from stable_baselines3.common.env_checker import check_env
# import evaluate as ev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ship_design
'''
description: 求解两个向量之间的距离，用l2范数
param {向量} a
param {向量} b
return {返回距离值，标量} np.sqrt(np.sum(c))
'''


def l2_norm(a, b):
    c = (a - b) ** 2
    return np.sqrt(np.sum(c))


class Env_One_Point(gym.Env):

    def __init__(self, arg, norm_data, ori_rank):
        """
        :param arg:
        """
        super(Env_One_Point, self).__init__()
        with open("env_design/env.json", 'r') as f:  # 读取环境配置文件
            env_conf = json.load(f)
        self.center = norm_data  # 圆心
        self.state = norm_data  # 初始状态
        self.arg = arg
        self.center_rank = ori_rank
        # self.fig = plt.figure()
        self.center_number = len(self.center)  # 圆心所有的维数
        self.radius = env_conf["radius_onepoint"]  # 邻域半径，超参数
        self.action_space = spaces.Discrete(int(self.center_number) * 2)  # 行为空间为圆心维数的2倍，离散型
        self.lisan = env_conf['lisan']  # 离散的程度，也就是加减的值
        self.actions = np.append(np.eye(self.center_number) * self.lisan, np.eye(self.center_number) * -self.lisan,
                                 axis=0)  # 行为设计，各维度加减0.01，应为超参数。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.center_number,),
                                            dtype=np.float64)  # 观测值空间，连续型
        self.weight=np.array([1000,10,10])
    def step(self, action):
        next_state = self.state + self.actions[action]*self.weight  # 下一个状态=前一个状态+行为
        print(next_state)
        with open("env_design/env.json", 'r') as f:  # 读取环境配置
            env_conf = json.load(f)

        if next_state[0]>0 and (next_state[1]>0 and next_state[1]<1) and (next_state[2]>0 and next_state[2]<1):
            value,rank,history=ship_design.DEtest(next_state)
            if self.center_rank > rank:
                # 如果得到的下一个状态分数比圆心的分数高过阈值，则搜索结束，得到所需要的解
                done = True
                self.state = next_state  # 将下一个状态付给当前状态
                self.center_rank=rank
                reward = env_conf['op_reward']  # 奖励值设置为10
                result=np.concatenate((value,[rank,history,next_state[0],next_state[1],next_state[2]]))
                pd.DataFrame(np.array([result])).to_csv("historyrecord.csv",float_format="%.2f", index=False, header=0,mode="a")

            else:  # 没有高过阈值，搜索没有结束，继续搜索
                done = False  # 没有完成
                # print("有进入")
                self.state = next_state  # 将下一个状态付给当前状态
                reward = env_conf['de_reward_step']  # 奖励值 设置为-0.1，每走一步都有一个较小的负奖励值，可以最小化步数。
            # TODO：还可以设置更严格的奖励值，如果下一状态的分数低于当前状态分数，也给予一个负的奖励值，也为超参数写入配置中
            # print(f"这一步是{self.state}")
        else:  # 下一状态超出邻域，搜索结束，在邻域内没有找到符合条件的解
            done = True  # 结束标志
            self.state = self.state  # 在邻域内的最后一个状态
            reward = -5  # 奖励函数为负值-5
            print("超限啦")
        info = {}  # 环境输出的额外的信息值，当前没有用到
        return self.state, reward, done, info

    def reset(self):  # 每个episode结束之后，将环境重置成初始状态
        self.state = self.center
        return self.state

    def render(self, mode="human"):  # 模型评估时的可视化函数，当前仅是输出状态值
        # TODO：可视化状态，可以实现
        # with open('out.csv','a') as f:
        #     f.write(f"{round(self.state[0],2)},{round(self.state[1],2)},{round(self.state[2],2)}\n")
        # print(f"当前状态是{self.state}")
        pass

    def seed(self, seed):  # 随机数种子
        np.random.seed(seed)


if __name__ == "__main__":
    env = Env_One_Point(arg=0,ori_rank=0,norm_data=np.array([0, 0]))
    print(env.action_space.n)
    check_env(env)  # 评估环境设计是否符合gym标准
