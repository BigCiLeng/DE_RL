import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def matrix_solve(result, se_index):
    result_list = []
    result_list.append(result['R_tatals_norm'])
    result_list.append(result['Bales_levels_norm'])
    result_list.append(result['Diameters_norm'])
    result_list = np.array(result_list).T
    # print(result_list)
    result_list = result_list[se_index]
    # print(result_list)
    sub_result = []
    for i in range(1, 9):
        sub_result.append(result_list[i - 1] - result_list[i])
    sub_result = np.array(sub_result)
    epsilon = np.random.random((sub_result.shape[0], 1)) * 0.001
    weights = np.linalg.lstsq(sub_result, epsilon, rcond=None)[0]
    # print(weights)
    return weights
