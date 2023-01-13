import pandas as pd
import numpy as np

def data_out(input_data):
    out = pd.DataFrame(input_data[:,35:input_data.shape[1]].T)
    tmp_colum = [i for i in range(1,11)]
    out.columns = tmp_colum
    tmp_index = []
    with open("data_output/name.txt",'r') as f:
        for i in f.readlines():
            tmp_index.append(i)
    out.index = tmp_index
    out.to_csv("data_output/out.csv",float_format="%.2f")

def rank_out(result):
    rank_list = [result["R_tatals_norm"],result['Bales_levels_norm'],result["Diameters_norm"],result['Displacements_norm'],result["GMs_norm"]]
    out = pd.DataFrame(np.array(rank_list))
    tmp_colum = [i for i in range(1,11)]
    out.columns = tmp_colum
    tmp_index = []
    with open("data_output/name.txt",'r') as f:
        for i in f.readlines():
            tmp_index.append(i)
    out.index = tmp_index
    out.to_csv("data_output/out.csv",float_format="%.2f")