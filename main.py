"""
Author: 孙凯威
Date: 2021-11-11 18:04:08
LastEditTime: 2021-11-15 21:03:20
LastEditors: 孙凯威
Description: 本地调试时使用的主函数代码，使用input_from_file.py作为输入。
FilePath: \ship_design\main_from_file.py
"""
import numpy as np
import torch
import data_input.input_from_file as di
import data_preprocess.preprocess as pre
import json
from multiprocessing import Process
import algorithm.test.ddqn_test as ddqn_test
import algorithm.test.doudqn_test as doudqn_test
import algorithm.test.dqn_test as dqn_test
import algorithm.doudqn as ddqn
import os
import data_preprocess.rank_eva as re
import data_preprocess.weights_matrix as wm

if __name__ == "__main__":
    with open('config.json', 'r') as f:  # 读配置文件，里面有最大进程数等
        conf = json.load(f)
    if (os.path.exists("data_output/data.csv")):
        os.remove("data_output/data.csv")
    if (os.path.exists("data_output/output.csv")):
        os.remove("data_output/output.csv")
    if (os.path.exists("data_output/evaluate.csv")):
        os.remove("data_output/evaluate.csv")

    process_list = []  # 进程列表，
    torch.multiprocessing.set_start_method('spawn')
    norm_data=[50,0.5,0.5]
    for i in range(conf['max_process']): 
        p = Process(target=ddqn.test_dqn, args=(0, norm_data, 0))
        p.start()
        process_list.append(p)
    for i in process_list:
        i.join()
