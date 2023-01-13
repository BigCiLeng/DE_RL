'''
Author: 孙凯威
Date: 2021-11-17 16:55:05
LastEditTime: 2021-11-17 18:45:34
LastEditors: Please set LastEditors
Description: 预处理代码，主要是归一化和排序函数
FilePath: /ship_design/data_preprocess/preprocess.py
'''
import numpy as np
import pandas as pd
import evaluate as ev

'''
description: 对待排序数组计算分数并排序
param {待排序数组} datas
return {排序完成索引序列} index_list
return {排序完成数组} data_set
return {得分排序数组} rank_list
'''

def data_solve(datas):
    datas_len = len(datas)  # 数组长度
    R_tatals = []  # 总阻力
    Bales_levels = []  # 耐波贝尔斯品级
    Diameters = []  # 回转直径
    Displacements = []  # 排水量
    GMs = []  # 初稳性高
    for data in datas:  # 循环数组中的数据
        data_len = len(data)  # 每个数据的维度
        # [船长 船宽 吃水 进流段长度 平行中体长度 去流段长度 浮心纵向位置 棱形系数
        # 半进流角 水线面系数 艉部纵向斜度 艏部倾斜角 球艏长度 球艏宽度 球艏基线以上高度
        # 球艏基线以下高度 艉部倾斜角 艉切点位置 #1舷侧倾斜角 #4舷侧倾斜角 #7舷侧倾斜角
        # #10舷侧倾斜角 #13舷侧倾斜角 #16舷侧倾斜角 #19舷侧倾斜角 #22舷侧倾斜角]
        temp = [data[0], data[1], data[3]] + list(data[35:40]) + [data[42], data[40], data[41]] + list(
            data[43:len(data)])
        temp1 = []
        for i in temp:
            temp1.append(round(i, 2))
        R_total, Bales_level, Diameter, Displacement, GM = ev.evaluate(temp1)
        R_tatals.append(R_total)
        Bales_levels.append(Bales_level)
        Diameters.append(Diameter)
        Displacements.append(Displacement)
        GMs.append(GM)
    # 归一化
    # print(R_tatals)
    result = {}
    R_tatals_norm, R_tatals_max, R_tatals_min = normalize(R_tatals)
    Bales_levels_norm, Bales_levels_max, Bales_levels_min = normalize(Bales_levels)
    Diameters_norm, Diameters_max, Diameters_min = normalize(Diameters)
    Displacements_norm, Displacements_max, Displacements_min = normalize(Displacements)
    GMs_norm, GMs_max, GMs_min = normalize(GMs)
    result['R_tatals_norm'] = R_tatals_norm
    result['R_tatals_max'] = R_tatals_max
    result['R_tatals_min'] = R_tatals_min
    result['Bales_levels_norm'] = Bales_levels_norm
    result['Bales_levels_max'] = Bales_levels_max
    result['Bales_levels_min'] = Bales_levels_min
    result['Diameters_norm'] = Diameters_norm
    result['Diameters_max'] = Diameters_max
    result['Diameters_min'] = Diameters_min
    result['Displacements_norm'] = Displacements_norm
    result['Displacements_max'] = Displacements_max
    result['Displacements_min'] = Displacements_min
    result['GMs_norm'] = GMs_norm
    result['GMs_max'] = GMs_max
    result['GMs_min'] = GMs_min
    return result


def rank(result, weights):
    rank_list = []  # 初始化得分数组
    for i in range(len(result['R_tatals_norm'])):
        rank = result['R_tatals_norm'][i]*weights[0]+result['Bales_levels_norm'][i]*weights[1]+result['Diameters_norm'][i]*weights[2]
        rank_list.append(rank)  # 将每个数据的得分加入得分数组
    index_list = np.argsort(rank_list)[::-1]  # 对得分数组的索引进行排序，递减排序
    # data_set = np.array([datas[index] for index in index_list])  # 将数组按照该索引序列进行排序
    rank_list = np.array([rank_list[index] for index in index_list])  # 将得分数组按照该索引序列排序
    return index_list, rank_list

def serank(result, weights):
    rank_list = []  # 初始化得分数组
    for i in range(len(result['R_tatals_norm'])):
        rank = result['R_tatals_norm'][i]*weights[0][0]+result['Bales_levels_norm'][i]*weights[1][0]+result['Diameters_norm'][i]*weights[2][0]
        rank_list.append(rank)  # 将每个数据的得分加入得分数组
    index_list = np.argsort(rank_list)[::-1]  # 对得分数组的索引进行排序，递减排序
    # data_set = np.array([datas[index] for index in index_list])  # 将数组按照该索引序列进行排序
    rank_list = np.array([rank_list[index] for index in index_list])  # 将得分数组按照该索引序列排序
    return index_list, rank_list

'''
description: 归一化数据
param {输入的数据数组，数据的分布差异比较大} datas
return {归一化后的数据} data_norm
return {数据各个特征中的最大值} data_max
return {数据各个特征中的最小值} data_min
'''


def normalize(datas):
    data_max = np.array([np.max(datas, axis=0)])+0.001  # 获取每一个维度的最大值
    data_min = np.array([np.min(datas, axis=0)])  # 获取每一个维度的最小值
    data_reduce = data_max - data_min  # 每个维度的极差值
    data_eav = datas - data_min  # 数据的每个维度距离最小值之间的差值
    data_norm = data_eav / data_reduce  # 归一化操作
    return data_norm, data_max, data_min


if __name__ == "__main__":
    # ds, rs = rank([[3, 4, 5, 6, 7], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    data_norm, data_max, data_min = normalize([[3, 4, 5, 6, 7], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
