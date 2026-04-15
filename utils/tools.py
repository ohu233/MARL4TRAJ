import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple
import matplotlib.image as mpimg
import warnings
from PIL import Image
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

MODE_LIST = ["GSD", "GG", "TS", "TG"]

def mapdata_to_modelmatrix(mapdata: dict, n_row, n_col) -> dict[str: list[list[int]]]:
    """
    Convert the mapdata to a matrix that can be used as input to the lower_model
    :param mapdata: dict, the mapdata
    县道、普铁、省道、高速收费站、高速、国道、高铁、火车站
    :return: dict, the matrix that can be used as input to the lower_model
    """
    modelmatrix = {"TG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "GG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "GSD": [[0 for _ in range(n_row)] for _ in range(n_col)],
                    "TS": [[0 for _ in range(n_row)] for _ in range(n_col)]
    }
    for k,v in mapdata.items():
        try:
            if v[4] & 1 == 1 or v[4] >>1 &1 == 1:
                modelmatrix['TG'][k[0]][k[1]] = 1
            if v[4] & 1 == 1 or v[4] >>6 &1 == 1 or v[4] >>1 &1 == 1:
                modelmatrix['TS'][k[0]][k[1]] = 1
            if v[4] >>3 & 1 == 1:
                modelmatrix['GG'][k[0]][k[1]] = 1
            if v[4] >>2 & 1 == 1 or v[4] >>5 & 1 == 1:
                modelmatrix['GSD'][k[0]][k[1]] = 1
        except:

            print('Input Data Out of Range: ',k,v, 'Map Size: ', n_row, n_col)

    return modelmatrix


def get_patch(modelmatrix, x, y, size=3)->list:
    """
    返回以 (x,y) 为中心的 size x size 展开补丁（包含中心），越界位置填 0
    """
    try:
        xmax = len(modelmatrix)
        ymax = len(modelmatrix[0])
    except:
        print('Input Data Out of Range When Getting Patch: ', type(modelmatrix), x, y)
        return [0 for _ in range(size*size)]

    patch = []
    n = (size - 1) // 2
    for dx in range(-n, n+1):
        for dy in range(-n, n+1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                patch.append(modelmatrix[nx][ny])
            else:
                patch.append(0)
    return patch
    

def state_to_vector(state: Dict, mode_list: List[str] = MODE_LIST) -> np.ndarray:
    """
    将 PathEnv 的 dict state 编码成 1D 向量：
    [initial(2), target(2), current(2), relative(2), mode_onehot(4), patch_flatten]
    """
    init_pos = np.array(state['current_position'], dtype=np.float32)
    target_pos = np.array(state['remaining_distance'], dtype=np.float32)
    cur_pos = np.array(state['previous_remaining_distance'], dtype=np.float32)
    rel_dis = np.array(state['total_distance'], dtype=np.float32)

    mode_onehot = np.zeros(len(mode_list), dtype=np.float32)
    current_mode = state['current_mode']
    if isinstance(current_mode, (list, tuple, np.ndarray)):
        for m in current_mode:
            if m in mode_list:
                mode_onehot[mode_list.index(m)] = 1.0
    else:
        if current_mode in mode_list:
            mode_onehot[mode_list.index(current_mode)] = 1.0

    patch = np.array(state['patch'], dtype=np.float32).reshape(-1)

    vec = np.concatenate([init_pos, target_pos, cur_pos, rel_dis, mode_onehot, patch], axis=0)
    return vec


def calculate_match_rate(traj_list: list, mapdata: np.ndarray) -> float:
    """
    计算轨迹点在路网上的匹配率（在路上的点数 / 总点数）
    越界点按“不在路上”处理，但仍计入总点数。
    """
    if not traj_list:
        return 0.0

    on_road = 0
    total = 0

    x_max, y_max = mapdata.shape[0], mapdata.shape[1]

    for p in traj_list:
        if p is None or len(p) < 2:
            continue

        x, y = int(round(p[0])), int(round(p[1]))
        total += 1

        if 0 <= x < x_max and 0 <= y < y_max:
            if mapdata[x, y] != 0:
                on_road += 1
        # else: 越界点默认 off-road（不加 on_road）
        
    if total == 0:
        return 0.0

    return on_road / total



def plt_multi_map(modes: List[str]):
    """
    在固定底图(js.jpg)上叠加指定 modes 的道路点，返回 fig, ax 供外部调用。
    """
    if modes is None:
        raise ValueError("modes 不能为空，例如 ['TG', 'GG']")

    valid_modes = {"TG", "GG", "GSD", "TS"}
    invalid = [m for m in modes if m not in valid_modes]
    if invalid:
        raise ValueError(f"不支持的 mode: {invalid}，可选: {sorted(valid_modes)}")

    with open(r"data\GridModesAdjacentRealworld.pkl", "rb") as f:
        mapdata = pickle.load(f)

    matrice = mapdata_to_modelmatrix(mapdata, 529, 564)

    mode_colors = {
        "TG": "orange",
        "GG": "blue",
        "GSD": "green",
        "TS": "red",
    }

    # 固定底图，不改
    bg_img = r"figur\jiangsu\js.jpg"

    # 用任意一个 mode 的矩阵拿尺寸
    ref_matrix = np.array(matrice["TG"])
    x_max, y_max = ref_matrix.shape[0], ref_matrix.shape[1]

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(
        mpimg.imread(bg_img),
        extent=[0, x_max, 0, y_max],
        aspect="equal",
        alpha=1
    )

    for mode in modes:
        matrix = np.array(matrice[mode])
        points = np.argwhere(matrix == 1)
        if points.size > 0:
            x = points[:, 0]
            y = points[:, 1]
            ax.scatter(
                x, y,
                s=5,      # 和 plt_js 一致
                c=mode_colors[mode],
                marker='+',
                linewidths=0.3,
                alpha=0.7
            )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('intermediate_fig.png')