'''
Author: 孙凯威
Date: 2021-11-17 16:55:05
LastEditTime: 2021-11-17 18:53:03
LastEditors: Please set LastEditors
Description: 从mat文件中读取数据，在本地自己调试时使用的方式
FilePath: /ship_design/data_input/input_from_file.py
'''
import pandas as pd
import numpy as np
import scipy.io as scio
'''
description: 从文件中读取数据
param {存放的路径} path
return {输出需要的数据数组} np.array(main_frame.values)
'''


def read_from_file(name='data_source/data1.mat'):
    path = 'data_source/'+name+'.mat'
    data = scio.loadmat(path)
    pri_dim = data[name][0][0]  # 主尺度参数：船长，船宽，型深，吃水，方形系数 5
    # print(len(pri_dim))
    func_pla = data[name][0][1]  # 功能布置参数 30
    # print(len(func_pla))
    contour = data[name][0][2]  # 型线
    temp = np.append(pri_dim, func_pla, axis=0)
    data = np.append(temp, contour, axis=0).T
    # print(data.shape)
    return data[:10]


if __name__ == '__main__':
    datas = read_from_file('data_source/data1.mat')
    print(datas)
