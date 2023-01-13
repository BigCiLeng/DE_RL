"""
Author: 孙凯威
Date: 2021-11-11 18:04:08
LastEditTime: 2021-11-15 21:03:20
LastEditors: 孙凯威
Description: 本地调试时使用的主函数代码，使用input_from_file.py作为输入。
FilePath: \ship_design\main_from_file.py
"""
import numpy as np
import pandas as pd
import torch
import data_input.input_from_file as di
import data_preprocess.preprocess as pre
import json
import multiprocessing
from multiprocessing import Process
import algorithm.test.ddqn_test as ddqn_test
import algorithm.test.doudqn_test as doudqn_test
import algorithm.test.dqn_test as dqn_test
import algorithm.ddqn as ddqn
import data_preprocess.rank_eva as re
import data_preprocess.weights_matrix as wm
import time

if __name__ == "__main__":
    with open('config.json', 'r') as f:  # 读配置文件，里面有最大进程数等
        conf = json.load(f)
    arg = {}
    weights = [1 / 3, 1 / 3, 1 / 3]
    # print("start reading data")
    start_time = time.time()
    input_datas = di.read_from_file(name=conf['read_from_file'])  # 调用input_from_file.py读取文件数据
    print(f"读取数据时间：{time.time() - start_time}")
    # print("reading data over")
    start_time = time.time()
    main_value = np.append(input_datas[:, 0:5], input_datas[:, 35:input_datas.shape[1]], axis=1)
    norm_datas, max_datas, min_datas = pre.normalize(main_value)  # 数据预处理，归一化
    print(f"数据归一化时间：{time.time() - start_time}")
    start_time = time.time()
    rank_result = pre.data_solve(input_datas)
    print(f"性能计算参数时间：{time.time() - start_time}")
    start_time = time.time()
    rank_index, rank_lists = pre.rank(rank_result, weights)  # 调用评分函数来排序
    print("ranking data over", rank_index)
    se_rank_index = re.se_rank()  # 获取专家希望的排序序列
    # print("start double ranking data")
    print(f"用户调整后的排序:{se_rank_index}")
    while True:
        new_weights = wm.matrix_solve(rank_result, se_rank_index)
        flag = True
        for i in new_weights:
            print(i[0])
            if i[0] < 0:
                flag = False
        if flag == True:
            break
    new_weights = new_weights * (1 / (np.sum(new_weights)))
    print("new_weights:", new_weights)
    print(f"权重优化参数时间：{time.time() - start_time}")
    norm_datas = norm_datas[se_rank_index]
    # print(norm_datas.shape)
    process_list = []  # 进程列表，
    arg['max_datas'] = max_datas
    arg['min_datas'] = min_datas
    arg['new_weights'] = new_weights
    arg['rank_result'] = rank_result
    torch.multiprocessing.set_start_method('spawn')
    start_time = time.time()
    for id,norm_data in enumerate(norm_datas[0:2, 0:]):  # 确定最大进程数，最后的输出结果个数也与此有关
        # p = Process(target=ddqn.test_dqn, args=(arg,norm_data,rank_lists[id]))
        p = Process(target=doudqn_test.test_dqn, args=(arg, norm_data, rank_lists[id]))
        p.start()
        process_list.append(p)
    for i in process_list:
        i.join()
    print(f"多参数优化时间：{time.time()-start_time}")
    data_out = pd.read_csv("data_output/data.csv", dtype=float, header=None).to_numpy()
    print(data_out.shape)
    data_main = data_out[:, 0:5].T  # 主尺度参数信息
    pd.DataFrame(data_main).to_csv("data_output/tmp_3_1.csv",float_format="%.2f", index=False, header=0)
    print(data_main.shape)
    data_xingxian = data_out[:, 5:28].T  # 型线参数
    pd.DataFrame(data_xingxian).to_csv("data_output/tmp_3_3.csv", float_format="%.2f", index=False, header=0)
    print(data_xingxian.shape)
    function_canshu = input_datas[:,5:35]
    data_function = function_canshu[se_rank_index[0:2]].T
    print(data_function.shape)
    pd.DataFrame(data_function).to_csv("data_output/tmp_3_2.csv", float_format="%.2f", index=False, header=0)
