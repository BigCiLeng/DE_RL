'''
description:获取专家排序的索引
return {排序完成索引数组} index_list
'''

import pandas as pd
import numpy as np
def se_rank():
    rank_modify = pd.read_csv("data_source/sequence.csv",header=None)
    rank_modify = np.array(rank_modify.iloc[:,0])
    index_list = list(rank_modify)  # 专家提供的排序序列5 2 0 6 1 9 3 7 8 4
    return index_list

if __name__=="__main__":
    se_rank()