import math
import pickle
from itertools import combinations

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.tools import mapdata_to_modelmatrix

Image.MAX_IMAGE_PIXELS = None

# 1) load data
with open(r"data\GridModesAdjacentRealworld.pkl", "rb") as f:
    mapdata = pickle.load(f)

matrice = mapdata_to_modelmatrix(mapdata, 529, 564)

# 2) base config
modes = ["TG", "GG", "GSD", "TS"]
mode_colors = {
    "TG": "orange",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}

bg_path = r"figur\jiangsu\js.jpg"
bg_img = mpimg.imread(bg_path)

# 3) precompute points for each mode
mode_points = {}
for mode in modes:
    matrix = matrice[mode]
    x, y = zip(*[(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == 1])
    mode_points[mode] = (x, y)

h = len(matrice[modes[0]])
w = len(matrice[modes[0]][0])

# 4) generate all non-empty combinations
all_combos = []
for r in range(1, len(modes) + 1):
    all_combos.extend(combinations(modes, r))

# 5) save one figure per combination (keep original style)
for combo in all_combos:
    fig, ax = plt.subplots(figsize=(20, 20))

    # same background style as your original code
    ax.imshow(bg_img, extent=[0, h, 0, w], aspect='equal', alpha=1)

    # overlay all modes in this combination
    for mode in combo:
        x, y = mode_points[mode]
        ax.scatter(x, y, s=1 / 100, alpha=1, c=mode_colors[mode], marker='o')

    ax.set_xlim(0, h)
    ax.set_ylim(0, w)
    ax.axis('off')

    combo_name = "+".join(combo)
    plt.tight_layout()
    plt.savefig(f'figure/{combo_name}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"Saved {len(all_combos)} combination figures.")