import os
import pickle
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

from utils.SoftActorCritic import (
    DiscreteSACAgent,
    SACConfig,
)
from utils.Environment import PathEnv
from utils.tools import state_to_vector, calculate_match_rate, plt_multi_map

# ========== 配置 ==========
traj_test = pd.read_csv('data/data_lower_test_filtered.csv')
EPISODES = len(traj_test)
MAX_STEPS = 300
MODEL_PATH = "FOV7entropy0.85Conv\sac_actor_ep5000_withConv.pth"
SAVE_DIR = None
FOV = 7
USE_CONV = True

# True: 测试时每个 episode 使用 row['mode']，不随机
# False: 保持环境原有随机 mode 采样
USE_ROW_MODE_FROM_DATA = False

# 底图相关
MAP_ROW, MAP_COL = 529, 564
BACKIMG_PATH = "figur/all_modes_js.png"

MODE_COLORS = {
    "TG": "orange",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}

FIGURE_DIR = "figure"
DEFAULT_BACKIMG_PATH = "figur/all_modes_js.png"
MODE_ORDER = ["TG", "GG", "GSD", "TS"]


def _normalize_modes(selected_mode, fallback_mode=None):
    if selected_mode is None:
        modes = [fallback_mode] if fallback_mode is not None else []
    elif isinstance(selected_mode, (list, tuple, np.ndarray, set)):
        modes = [str(m).strip() for m in selected_mode]
    else:
        modes = [str(selected_mode).strip()]

    valid = [m for m in modes if m in MODE_ORDER]
    if not valid and fallback_mode in MODE_ORDER:
        valid = [fallback_mode]

    # 按固定顺序排序，和 figure 文件命名保持一致
    valid = sorted(set(valid), key=lambda x: MODE_ORDER.index(x))
    return valid


def _load_mode_background(selected_mode, fallback_mode=None, cache=None):
    if cache is None:
        cache = {}

    modes = _normalize_modes(selected_mode, fallback_mode=fallback_mode)
    if not modes:
        key = ("__default__",)
    else:
        key = tuple(modes)

    if key in cache:
        return cache[key]

    if modes:
        fig_name = "+".join(modes) + ".png"
        candidate = os.path.join(FIGURE_DIR, fig_name)
        if os.path.exists(candidate):
            cache[key] = mpimg.imread(candidate)
            return cache[key]

    # 回退到底图
    cache[key] = mpimg.imread(DEFAULT_BACKIMG_PATH)
    print(f"[WARN] 背景图不存在，使用默认底图: modes={modes}")
    return cache[key]

def mode_legend_handles():
    return [
        Line2D([0], [0], color="orange", lw=2, label="TG"),
        Line2D([0], [0], color="blue", lw=2, label="GG"),
        Line2D([0], [0], color="green", lw=2, label="GSD"),
        Line2D([0], [0], color="red", lw=2, label="TS"),
    ]

def load_env(traj_df, use_row_mode_from_data: bool = False, fov: int = 7):
    """和训练时保持一致的 PathEnv 配置，只是用传入的 traj_df."""
    with open('data/GridModesAdjacentRealworld.pkl', 'rb') as f:
        mapdata = pickle.load(f)

    env = PathEnv(
        train_mode=not use_row_mode_from_data,  # 开关为 True 时关闭随机
        mapdata=mapdata,
        traj=traj_df,
        FOV=fov,
        distance_threshold=1.0,
    )
    return env


def load_agent(env, model_path: str, use_conv: bool = True):
    """创建 DiscreteSACAgent，并加载 actor 权重。"""
    cfg = SACConfig(device="cpu")
    device = torch.device(cfg.device)

    agent = DiscreteSACAgent(vec_dim=12, fov=env.FOV, action_dim=4,
                              cfg=cfg, use_conv=use_conv)
    state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    env.traj_cnt = 0
    return agent


def run_eval_with_plots(env, agent, traj_df, episodes: int,
                        max_steps: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    all_trajs = []          # 存实际坐标轨迹
    ep_rewards = []
    ep_success_flags = []
    traj_records = []

    x_min, y_min, x_max, y_max = 1e9, 1e9, -1e9, -1e9
    bg_cache = {}

    current_id = None
    id_buffer = []

    def _flush_id_buffer():
        if id_buffer:
            _plot_combined_for_id(id_buffer, current_id, save_dir)
            id_buffer.clear()

    for ep in range(episodes):
        # 当前使用的测试数据行
        row_idx = ep
        if row_idx >= len(traj_df):
            break
        row = traj_df.iloc[row_idx]
        row_id = str(row['ID'])

        # ID 变化时，立即生成上一个 ID 的聚合图
        if current_id is not None and row_id != current_id:
            _flush_id_buffer()
        current_id = row_id

        mode = str(row['mode']).strip()

        if USE_ROW_MODE_FROM_DATA:
            # 在 reset 前设置 selected_mode，reset 会据此生成 multi_mapdata
            env.selected_mode = np.array([mode], dtype=object)

        state = env.reset()

        # ========= 用“当前测试数据的起点 xy”作为偏移 =========
        # 假设列名为 locx_o / locy_o（和你 dual 脚本一致）
        delta = np.array([row['locx_o'], row['locy_o']], dtype=float)
        end_xy = np.array([row['locx_d'], row['locy_d']], dtype=float)  # 终点使用 row
        mode = row['mode']
        # =================================================

        grid_pos = np.array(state['current_position'], dtype=float)  # 栅格坐标
        actual_pos = grid_pos + delta                                # 实际坐标（加偏移）
        traj = [actual_pos.copy()]
        total_reward = 0.0
        success_flag = 0

        for t in range(max_steps):
            s_vec = state_to_vector(state)
            a = agent.select_action(s_vec, evaluate=True)
            next_state, r, done, success = env.step(int(a))

            grid_pos = np.array(next_state['current_position'], dtype=float)
            actual_pos = grid_pos + delta
            traj.append(actual_pos.copy())

            total_reward += float(r)
            if success:
                success_flag = 1

            state = next_state

            x_min = min(x_min, actual_pos[0])
            x_max = max(x_max, actual_pos[0])
            y_min = min(y_min, actual_pos[1])
            y_max = max(y_max, actual_pos[1])

            if done:
                break

        traj_arr = np.array(traj)
        print(calculate_match_rate(traj_arr.tolist(), env.multi_mapdata))

        item = {
            "traj": traj_arr,
            "mode": str(mode),
            "id": str(row['ID']),
            "start_xy": traj_arr[0].copy(),
            "end_xy": end_xy.copy(),
        }
        all_trajs.append(item)
        id_buffer.append(item)
        ep_rewards.append(total_reward)
        ep_success_flags.append(success_flag)

        traj_list = [[float(p[0]), float(p[1])] for p in traj_arr]
        traj_records.append({
            "episode": ep,
            "order": int(row['order']) if 'order' in row else int(row_idx),
            "reward": float(total_reward),
            "success": int(success_flag),
            "match": calculate_match_rate(traj_arr.tolist(), env.multi_mapdata),
            "mode": 1 if mode in env.selected_mode else 0,
            "traj": json.dumps(traj_list, ensure_ascii=False),
        })

        # ====== 画当前 episode 的带底图轨迹（用 actual_pos） ======
        try:
            mode_str = str(mode).strip()
            episode_bg = _load_mode_background(
                selected_mode=getattr(env, "selected_mode", None),
                fallback_mode=mode_str,
                cache=bg_cache,
            )

            height, width = episode_bg.shape[0], episode_bg.shape[1]
            ratio = ((height / MAP_ROW) * (width / MAP_COL)) ** 0.5
            xs = traj_arr[:, 0]
            ys = traj_arr[:, 1]
            x_min_local, x_max_local = xs.min() - 2, xs.max() + 2
            y_min_local, y_max_local = ys.min() - 2, ys.max() + 2

            x_min_idx = int(max(0, x_min_local * ratio))
            x_max_idx = int(min(width,  x_max_local * ratio))
            y_min_idx = int(max(0, y_min_local * ratio))
            y_max_idx = int(min(height, y_max_local * ratio))

            sliced_img = episode_bg[
                height - y_max_idx: height - y_min_idx,
                x_min_idx:x_max_idx
            ]

            plt.figure(figsize=(6, 5))
            plt.imshow(
                sliced_img,
                extent=[x_min_local, x_max_local, y_min_local, y_max_local],
                alpha=0.5
            )

            mode_str = str(mode).strip()
            traj_color = MODE_COLORS.get(mode_str, "C0")

            plt.plot(traj_arr[:, 0], traj_arr[:, 1],
                     marker='o', markersize=2, color=traj_color)

            # start
            plt.scatter(traj_arr[0, 0], traj_arr[0, 1],
            c=traj_color, marker='o', s=100,
            edgecolors='red', linewidths=2, label='start')

            # end
            plt.scatter(end_xy[0], end_xy[1],
            c=traj_color, marker='x', s=120,
            linewidths=2, label='end')

            # agent end
            agent_end = traj_arr[-1]
            plt.scatter(agent_end[0], agent_end[1],
            c='black', marker='^', s=60, linewidths=1.5, label='agent_end')

            start_handle = Line2D([0], [0], marker='o', color='w', label='start',
                      markerfacecolor=traj_color, markeredgecolor='red',
                      markersize=8, linewidth=0)
            end_handle = Line2D([0], [0], marker='x', color=traj_color, label='end',
                    markersize=8, linewidth=0)

            plt.legend(handles=mode_legend_handles() + [start_handle, end_handle], loc='best')

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(
                f"Ep {ep}, "
                f"Succ={success_flag == 1}, Mode={mode}, "
                f"Selected Mode={env.selected_mode}"
            )
            plt.grid(True)
            plt.legend()

            # 起终点坐标（实际坐标）
            sx, sy = traj_arr[0, 0], traj_arr[0, 1]
            ex, ey = end_xy[0], end_xy[1]

            # 文件名里避免小数点过长/非法字符，统一保留2位并把负号保留
            filename = (
                f"ep_{ep:04d}"
                f"_succ_{success_flag}"
                f"_match_{calculate_match_rate(traj_arr.tolist(), env.multi_mapdata):.2f}"
                # f"_S({sx:.2f},{sy:.2f})"
                # f"_E({ex:.2f},{ey:.2f})"
                ".png"
            )

            # Windows 下文件名不建议含冒号等符号；这里是安全的
            ep_path = os.path.join(save_dir, filename)
            # plt.savefig(ep_path, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"[Episode {ep}] reward={total_reward:.3f}, success={success_flag}, saved: {ep_path}")
            if ep % 100 == 0:
                print(f"  Current success rate: {np.mean(ep_success_flags) * 100:.2f}%")
        except Exception as e:
            print(f"Error plotting episode {ep}: {e}")

    # 最后一个 ID 的聚合图
    _flush_id_buffer()

    # ====== 保存 CSV ======
    df = pd.DataFrame(traj_records)
    csv_path = os.path.join(save_dir, "traj_records.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved traj records CSV to: {csv_path}")

    print("=" * 60)
    print("Avg reward   : {:.3f}".format(np.mean(ep_rewards)))
    print("Success rate : {:.2f}%".format(np.mean(ep_success_flags) * 100))
    print("=" * 60)

    return all_trajs


def _plot_combined_for_id(items, tid, save_dir):
    """为单个 ID 生成轨迹聚合图。"""
    combined_dir = os.path.join(save_dir, "combined_by_id")
    os.makedirs(combined_dir, exist_ok=True)

    # 用该 ID 实际涉及的模式作为底图
    used_modes = list({item["mode"] for item in items})
    bg_img = _load_mode_background(selected_mode=used_modes)
    height, width = bg_img.shape[0], bg_img.shape[1]
    ratio = ((height / MAP_ROW) * (width / MAP_COL)) ** 0.5

    # 收集所有点以确定整体范围
    all_xs = []
    all_ys = []
    for item in items:
        all_xs.extend(item["traj"][:, 0].tolist())
        all_ys.extend(item["traj"][:, 1].tolist())

    x_min = min(all_xs) - 2
    x_max = max(all_xs) + 2
    y_min = min(all_ys) - 2
    y_max = max(all_ys) + 2

    x_min_idx = int(max(0, x_min * ratio))
    x_max_idx = int(min(width, x_max * ratio))
    y_min_idx = int(max(0, y_min * ratio))
    y_max_idx = int(min(height, y_max * ratio))

    sliced_img = bg_img[
        height - y_max_idx : height - y_min_idx,
        x_min_idx : x_max_idx,
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        sliced_img,
        extent=[x_min, x_max, y_min, y_max],
        alpha=0.5,
    )

    # 逐段绘制
    prev_end = None
    for item in items:
        traj = item["traj"]
        mode_str = item["mode"]
        color = MODE_COLORS.get(mode_str, "C0")

        ax.plot(
            traj[:, 0], traj[:, 1],
            marker="o", markersize=2, color=color, linewidth=1.5,
        )

        # 虚线连接上一段终点和本段起点
        if prev_end is not None:
            seg_start = item["start_xy"]
            ax.plot(
                [prev_end[0], seg_start[0]],
                [prev_end[1], seg_start[1]],
                linestyle="--", color="gray", linewidth=0.8, alpha=0.7,
            )
        prev_end = item["end_xy"]

    # 整体起点：第一个路段起点
    first_start = items[0]["start_xy"]
    ax.scatter(
        first_start[0], first_start[1],
        c=MODE_COLORS.get(items[0]["mode"], "C0"),
        marker="o", s=120, edgecolors="red", linewidths=2,
        zorder=5, label="start",
    )

    # 整体终点：最后一个路段终点
    last_end = items[-1]["end_xy"]
    ax.scatter(
        last_end[0], last_end[1],
        c=MODE_COLORS.get(items[-1]["mode"], "C0"),
        marker="x", s=120, linewidths=2, zorder=5, label="end",
    )

    # 图例：mode颜色 + start/end
    handles = mode_legend_handles()
    start_handle = Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="gray", markeredgecolor="red",
        markersize=8, linewidth=0, label="start",
    )
    end_handle = Line2D(
        [0], [0], marker="x", color="gray",
        markersize=8, linewidth=0, label="end",
    )
    ax.legend(handles=handles + [start_handle, end_handle], loc="best")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"ID={tid}, segments={len(items)}")
    ax.grid(True)

    save_path = os.path.join(combined_dir, f"{tid}.png")
    # fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Combined] ID={tid}, {len(items)} segments → {save_path}")


if __name__ == "__main__":
    if SAVE_DIR is None:
        base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        SAVE_DIR = os.path.join(base + "_test")

    traj_test = pd.read_csv('data\\data_lower_test_filtered.csv')

    env = load_env(traj_test, use_row_mode_from_data=USE_ROW_MODE_FROM_DATA, fov=FOV)
    agent = load_agent(env, MODEL_PATH, use_conv=USE_CONV)

    run_eval_with_plots(
        env, agent,
        traj_df=traj_test,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        save_dir=SAVE_DIR,
    )