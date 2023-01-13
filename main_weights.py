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
            if i[0]<0:
                flag = False
        if flag == True:
            break
    print(f"权重优化参数时间：{time.time() - start_time}")
    new_weights_to1 = new_weights*(1/(np.sum(new_weights)))
    pd.DataFrame(new_weights_to1).to_csv("data_output/weight.csv",index=False,header=False,float_format="%.4f")
