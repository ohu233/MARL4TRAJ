import pickle
import numpy as np
import pandas as pd
import torch

from utils.SoftActorCritic import DiscreteSACAgent, SACConfig
from utils.tools import mapdata_to_modelmatrix, get_patch, state_to_vector, calculate_match_rate


# global variables
dxdy_dict = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
modelist = ['GSD', 'GG', 'TS', 'TG']

with open('data/GridModesAdjacentRealworld.pkl', 'rb') as f:
    mapdata = pickle.load(f)

class PathEnv:
    '''
    Train：（单独训练）在随机扰动的Mode选择下，进行路径恢复，输出Path
    Test：（协同部署）接收ModeAgent传递的Mode
    '''
    def __init__(self, 
                 selected_mode: np.ndarray=None, 
                 train_mode: bool=True, 
                 curriculum_mode: bool=True,
                 mapdata: dict=None, 
                 traj: pd.DataFrame=None,
                 FOV: int=5,
                 distance_threshold: float=1.0,
                 ):
        
        self.selected_mode = selected_mode
        self.train_mode = train_mode
        self.curriculum_mode = curriculum_mode
        self.mapdata = mapdata
        self.traj = traj
        self.traj_cnt = 0
        self.FOV = FOV
        self.distance_threshold = distance_threshold
        realmap_row = 529   # 326
        realmap_col = 564   # 364
        self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.node_memory = set()
        self.curriculum_stage = 0
        self.min_mode_count = 1
        self.max_mode_count = len(modelist)
        if self.selected_mode is None:
            self.selected_mode = np.array(modelist)

    def reset(self):

        self.step_cnt = 0
    
        # 计算当前轨迹索引
        current_traj_idx  = self.traj_cnt % len(self.traj)

        if self.train_mode:
            max_modes = self.max_mode_count if self.curriculum_mode else len(modelist)
            max_modes = max(1, min(max_modes, len(modelist)))
            min_modes = max(1, min(self.min_mode_count, max_modes))
            num_modes = np.random.randint(min_modes, max_modes + 1)
            self.selected_mode = np.random.choice(modelist, size=num_modes, replace=False)
        else:
            if self.selected_mode is None or len(self.selected_mode) == 0:
                self.selected_mode = np.array(modelist)

        self.multi_mapdata = np.zeros_like(self.mapdata[self.selected_mode[0]])
        for mode in self.selected_mode:
            self.multi_mapdata += self.mapdata[mode]

        self.locx_start = float(self.traj.loc[current_traj_idx, 'locx_o'])
        self.locy_start = float(self.traj.loc[current_traj_idx, 'locy_o'])
        self.locx_end = float(self.traj.loc[current_traj_idx, 'locx_d'])
        self.locy_end = float(self.traj.loc[current_traj_idx, 'locy_d'])

        self.neighbor = get_patch(self.multi_mapdata, self.locx_start, self.locy_start, size=3)

        self.max_step = max(1, int((abs(self.locx_start - self.locx_end) + abs(self.locy_start - self.locy_end)) * 6))

        self.traj_cnt += 1

        #引入Node Memory
        self.node_memory = dict()

        self.state = {'current_position':np.array([0, 0]),  # 当前位置偏移
                      'remaining_distance':np.array([self.locx_end - self.locx_start, self.locy_end - self.locy_start]), # 剩余距离向量
                      'previous_remaining_distance':np.array([self.locx_end - self.locx_start, self.locy_end - self.locy_start]), # 上一步剩余距离
                      'total_distance':np.array([self.locx_end - self.locx_start, self.locy_end - self.locy_start]), # 总距离（定值）
                      'current_mode':self.selected_mode,    # 当前模式（定值）
                      'patch':(get_patch(self.multi_mapdata, self.locx_start, self.locy_start, size=self.FOV)), # 邻域信息
                      'visit_count': 0
                      }
        
        return self.state

    def split_traj_by_distance(self, num_stages: int = 4):
        if self.traj is None or len(self.traj) == 0:
            return [self.traj]

        required_cols = {'locx_o', 'locy_o', 'locx_d', 'locy_d'}
        if not required_cols.issubset(self.traj.columns):
            missing = required_cols - set(self.traj.columns)
            raise ValueError(f"Trajectory dataframe missing required columns: {missing}")

        df = self.traj.copy().reset_index(drop=True)
        # 与环境一致，使用曼哈顿距离做课程分段。
        df['_dist'] = (df['locx_d'] - df['locx_o']).abs() + (df['locy_d'] - df['locy_o']).abs()

        bins = min(num_stages, max(1, int(df['_dist'].nunique())))
        if bins == 1:
            return [df.drop(columns=['_dist']).reset_index(drop=True)]

        sid = pd.qcut(df['_dist'], q=bins, labels=False, duplicates='drop')
        df['_sid'] = sid.astype(int)

        stage_trajs = []
        for stage_id in sorted(df['_sid'].unique().tolist()):
            stage_df = df[df['_sid'] == stage_id].drop(columns=['_dist', '_sid']).reset_index(drop=True)
            stage_trajs.append(stage_df)

        return stage_trajs

    def set_curriculum_stage(self, stage_idx: int, traj_subset: pd.DataFrame = None, max_mode_count: int = 4):
        self.curriculum_stage = int(stage_idx)
        self.max_mode_count = max(1, min(int(max_mode_count), len(modelist)))
        if traj_subset is not None:
            self.traj = traj_subset.reset_index(drop=True)
            self.traj_cnt = 0

    def set_mode_sampling_range(self, min_mode_count: int = 1, max_mode_count: int = 4):
        self.min_mode_count = max(1, min(int(min_mode_count), len(modelist)))
        self.max_mode_count = max(1, min(int(max_mode_count), len(modelist)))
        if self.min_mode_count > self.max_mode_count:
            self.min_mode_count = self.max_mode_count
    
    def calculate_reward(self, reward, prev_dist, curr_dist, neighbor, action):
        '''
        改进后的奖励函数
        '''
        #道路奖励：检查是否在道路上
        # neighbor索引映射：0->(1,0)->idx 7 (下); 1->(0,1)->idx 5 (右); 2->(-1,0)->idx 1 (上); 3->(0,-1)->idx 3 (左)
        action_to_idx = {0: 7, 1: 5, 2: 1, 3: 3}
        is_on_road = neighbor[action_to_idx[action]] != 0
        dist_change = prev_dist - curr_dist        
        '''        
        if is_on_road:
            reward += 1  # 走在路上给予正向奖励
        else:
            reward -= 3 # 走到非道路区域给予惩罚

        reward += dist_change * 1  # 系数可调，鼓励靠近
        '''

        if is_on_road:
            reward += 8
            if dist_change >= 0:
                reward += 3

        else:
            reward -= 10
            if dist_change >= 0:
                reward -= 2
            else:
                reward -= 4


        # 3. 时间/步数惩罚：鼓励尽快到达
        # reward -= 0.3

        return reward

    def step(self, action: int):
        '''
        采取动作，计算奖励，更新状态
        '''
        success = 0
        reward = 0.0
        done = False
        self.step_cnt += 1

        # 计算移动前的距离（Manhattan）
        prev_dist = abs(self.state['remaining_distance'][0]) + abs(self.state['remaining_distance'][1])
        
        # 更新位置偏移
        self.state['current_position'] = (self.state['current_position'][0] + dxdy_dict[action][0], self.state['current_position'][1] + dxdy_dict[action][1])

  
        # 更新节点记忆
        if self.state['current_position'] in self.node_memory:
            self.node_memory[self.state['current_position']] += 1
        else:
            self.node_memory[self.state['current_position']] = 1

        visit_count = self.node_memory.get(self.state['current_position'])
        self.state['visit_count'] = visit_count
        reward -= visit_count * 2

        # 更新绝对坐标
        self.locx_start += dxdy_dict[action][0]
        self.locy_start += dxdy_dict[action][1]

        # 更新上一步剩余距离向量
        self.state['previous_remaining_distance'] = self.state['remaining_distance']

        # 更新剩余距离向量
        self.state['remaining_distance'] = (self.locx_end - self.locx_start, self.locy_end - self.locy_start)

        # 计算移动后的距离
        curr_dist = abs(self.state['remaining_distance'][0]) + abs(self.state['remaining_distance'][1])
        
        # 计算奖励
        reward = self.calculate_reward(reward, prev_dist, curr_dist, self.neighbor, action)
        
        # 更新neighbor为新位置的邻域信息
        self.neighbor = get_patch(self.multi_mapdata, self.locx_start, self.locy_start, size=3)
        self.state['patch'] = get_patch(self.multi_mapdata, self.locx_start, self.locy_start, size=self.FOV)

        # 判断done:涉及reward更新
        if curr_dist <= self.distance_threshold:
            done = True
            success = 1
            reward += 90
        elif self.step_cnt >= self.max_step:
            done = True
            reward -= 30
        else:
            done = False

        return self.state, reward, done, success



class ModeEnv:
    # TODO:重写:输入为同一ID的一批数据
    """
    Train: 选择 mode 组合（4bit），调用已训练 PathAgent 回放路径，输出匹配指标与奖励
    Test:  同样流程，但关闭扰动，使用评估动作
    """
    def __init__(
        self,
        model_path: str,
        mapdata,
        traj: pd.DataFrame,
        train_mode: bool = True,
        fov: int = 5,
        distance_threshold: float = 1.0,
    ):
        self.model_path = model_path
        self.mapdata = mapdata
        self.traj = traj
        self.train_mode = train_mode
        self.fov = fov
        self.distance_threshold = distance_threshold
        self.max_mode_steps = 50

        self.no_change_patience = 5      # 连续多少步不变就提前结束
        self.no_change_streak = 0

        self.row = 529
        self.col = 564
        self.traj_cnt = 0
        self.current_row = None
        self.mode_maps = mapdata_to_modelmatrix(self.mapdata, self.row, self.col)
        self.mode_speed_stats = self._build_mode_speed_stats()

        traj_dummy = pd.DataFrame({"locx_o": [0], "locy_o": [0], "locx_d": [1], "locy_d": [1]})
        path_env = PathEnv(
            train_mode=False,
            selected_mode=np.array(modelist),
            mapdata=self.mapdata,
            traj=traj_dummy,
            FOV=self.fov,
            distance_threshold=0.0,
        )

        s0 = path_env.reset()
        s0_vec = state_to_vector(s0)
        state_dim = s0_vec.shape[0]
        action_dim = 4

        cfg = SACConfig()
        device = torch.device(cfg.device)

        path_agent = DiscreteSACAgent(state_dim, action_dim, cfg)
        state_dict = torch.load(self.model_path, map_location=device)
        path_agent.actor.load_state_dict(state_dict)
        path_agent.actor.eval()
        self.path_agent = path_agent

    def _build_mode_speed_stats(self):
        """
        基于轨迹数据估计每个 mode 的速度分布参数（均值/标准差）。
        若某 mode 样本不足，则回退到全局统计量。
        """
        stats = {}
        fallback_mean = 0.0
        fallback_std = 1.0

        if self.traj is None or len(self.traj) == 0:
            return {m: {"mean": fallback_mean, "std": fallback_std} for m in modelist}

        if "mode" not in self.traj.columns or "velocity" not in self.traj.columns:
            return {m: {"mean": fallback_mean, "std": fallback_std} for m in modelist}

        mode_col = self.traj["mode"].astype(str).str.strip()
        vel_col = pd.to_numeric(self.traj["velocity"], errors="coerce")

        global_vel = vel_col.dropna()
        if len(global_vel) > 0:
            fallback_mean = float(global_vel.mean())
            global_std = float(global_vel.std(ddof=0))
            if np.isfinite(global_std) and global_std > 1e-6:
                fallback_std = global_std

        for m in modelist:
            v = vel_col[mode_col == m].dropna().to_numpy(dtype=np.float32)
            if len(v) == 0:
                mu = fallback_mean
                std = fallback_std
            else:
                mu = float(np.mean(v))
                std = float(np.std(v))
                if (not np.isfinite(std)) or std <= 1e-6:
                    std = fallback_std

            stats[m] = {"mean": mu, "std": std}

        return stats

    def _speed_deviation_reward(self, cur_mask, velocity: float) -> float:
        """
        速度奖励：按当前所选 mode 对应高斯分布打分。
        score = exp(-0.5 * z^2), z=(v-mu)/sigma
        再线性映射到 [-1, 1]，偏离越大惩罚越强。
        """
        selected_modes = self._mask_to_modes(cur_mask)
        if len(selected_modes) == 0:
            return 0.0

        rewards = []
        for m in selected_modes:
            st = self.mode_speed_stats.get(m)
            if st is None:
                continue

            mu = float(st.get("mean", 0.0))
            sigma = float(st.get("std", 1.0))
            sigma = max(sigma, 1e-6)

            z = (float(velocity) - mu) / sigma
            gaussian_score = float(np.exp(-0.5 * (z ** 2)))
            rewards.append(2.0 * gaussian_score - 1.0)

        if len(rewards) == 0:
            return 0.0

        return float(np.mean(rewards))

    def _mask_to_modes(self, mask):
        return [modelist[i] for i, v in enumerate(mask) if int(v) == 1]

    def _default_mode_mask(self):
        return [1, 1, 1, 1]

    def _infer_init_mode_mask(self, idx: int):
        """
        初始化 mode 状态：
        - 若上一条样本与当前样本 ID 相同，则用上一条终点(locx_d, locy_d)所在模式作为初始mask
        - 若无上一条样本/ID不一致/无法定位模式，则默认四模式全开
        """
        if len(self.traj) <= 1 or idx <= 0:
            return self._default_mode_mask()

        if "ID" not in self.traj.columns:
            return self._default_mode_mask()

        cur_row = self.traj.iloc[idx]
        prev_row = self.traj.iloc[idx - 1]

        cur_id = str(cur_row.get("ID", "")).strip()
        prev_id = str(prev_row.get("ID", "")).strip()
        if cur_id == "" or prev_id == "" or cur_id != prev_id:
            return self._default_mode_mask()

        if ("locx_d" not in self.traj.columns) or ("locy_d" not in self.traj.columns):
            return self._default_mode_mask()

        try:
            x = int(round(float(prev_row["locx_d"])))
            y = int(round(float(prev_row["locy_d"])))
        except Exception:
            return self._default_mode_mask()

        mode_maps = self.mode_maps
        if len(mode_maps) == 0:
            return self._default_mode_mask()

        first_map = np.asarray(mode_maps[modelist[0]])
        x_max, y_max = first_map.shape[0], first_map.shape[1]
        if not (0 <= x < x_max and 0 <= y < y_max):
            return self._default_mode_mask()

        mask = []
        for m in modelist:
            mode_grid = np.asarray(mode_maps[m])
            mask.append(1 if mode_grid[x, y] != 0 else 0)

        if int(np.sum(mask)) == 0:
            return self._default_mode_mask()

        return mask
    
    def _action_to_index(self, action):
        """
        动作语义：
        0 -> 不变
        1~4 -> 翻转对应 mode bit（1->GSD, 2->GG, 3->TS, 4->TG）
        """
        if not np.isscalar(action):
            arr = np.asarray(action).reshape(-1)
            if arr.shape[0] in (4, 5):
                # 若上层偶尔传 onehot/score，取 argmax 兼容
                a = int(np.argmax(arr))
            else:
                raise ValueError(f"action must be scalar in [0,4] or len-5 array, got shape {arr.shape}")
        else:
            a = int(action)

        if a < 0 or a > 4:
            raise ValueError(f"action id out of range: {a}, expected [0, 4]")
        return a

    def _flip_one_bit(self, prev_mask, action):
        m = np.asarray(prev_mask, dtype=np.int64).copy()
        idx = self._action_to_index(action)

        # idx=0 表示不变，允许策略在当前组合上停留以便收敛
        if idx == 0:
            return m

        bit_idx = idx - 1
        m[bit_idx] = 1 - m[bit_idx]

        # 不允许全关
        if int(m.sum()) == 0:
            m[bit_idx] = 1

        return m

    def _run_PathMode(self, selected_modes):
        traj_one = self.current_row.reset_index(drop=True)  # 当前数据

        # 实例化环境
        env = PathEnv(
            train_mode=False,
            selected_mode=np.array(selected_modes),
            mapdata=self.mapdata,
            traj=traj_one,
            FOV=self.fov,
            distance_threshold=self.distance_threshold,
        )

        s = env.reset()
        traj_points = [(env.locx_start, env.locy_start)]    # 记录途经点
        steps = 0
        trans_times = 0     # 记录模式是否发生转换
        success = 0
        done = False

        while not done:
            s_vec = state_to_vector(s)
            a = self.path_agent.select_action(s_vec, evaluate=True)

            s, _, done, succ = env.step(int(a))
            nx, ny = env.locx_start, env.locy_start
            traj_points.append((nx, ny))

            steps += 1
            success = int(succ)

        path_len = float(steps)

        # 计算 trans_times：严格模式（无 mode 也算一种 mode）
        trans_times = 0
        candidate_modes = None
        NO_MODE = "__NO_MODE__"

        x_max, y_max = env.multi_mapdata.shape[0], env.multi_mapdata.shape[1]

        for p in traj_points:
            if p is None or len(p) < 2:
                continue

            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < x_max and 0 <= y < y_max):
                continue

            # 当前点在 selected_modes 中可用的 mode 集合
            curr_modes = {m for m in selected_modes if np.asarray(env.mapdata[m])[x, y] != 0}

            # 严格：没有任何 mode 时，视作一种独立 mode
            if not curr_modes:
                curr_modes = {NO_MODE}

            if candidate_modes is None:
                candidate_modes = set(curr_modes)
                continue

            candidate_modes &= curr_modes

            if len(candidate_modes) == 0:
                trans_times += 1
                candidate_modes = set(curr_modes)


        # 在已选择的混合路网的综合匹配度
        multi_match_rate = float(calculate_match_rate(traj_points, np.asarray(env.multi_mapdata)))

        # 在“当前选择的模式集合”内做占比统计：
        selected_set = set(selected_modes)
        mode_scores = {m: 0.0 for m in modelist}

        # 分母用总步数（动作数）
        total_points = float(max(len(traj_points), 1))

        if len(traj_points) > 0 and len(selected_modes) > 0:
            ref_map = np.asarray(env.multi_mapdata)
            x_max, y_max = ref_map.shape[0], ref_map.shape[1]

            for p in traj_points:
                if p is None or len(p) < 2:
                    continue

                x, y = int(round(p[0])), int(round(p[1]))
                if not (0 <= x < x_max and 0 <= y < y_max):
                    continue

                for m in selected_modes:
                    if np.asarray(env.mapdata[m])[x, y] != 0:
                        mode_scores[m] += 1.0

        match_rate = []
        for m in modelist:
            if m in selected_set:
                match_rate.append(float(mode_scores[m] / total_points))
            else:
                match_rate.append(0.0)

        return match_rate, multi_match_rate, success, steps, path_len, trans_times

    def reset(self):
        idx = self.traj_cnt % len(self.traj)
        self.current_row = self.traj.iloc[[idx]].copy()
        self.traj_cnt += 1
        self.step_cnt = 0
        self.finish = False
        self.no_change_streak = 0

        init_mode_mask = self._infer_init_mode_mask(idx)

        self.state = {
            "previous":{
                "mode": [0, 0, 0, 0],
                "match_rate": [0.0, 0.0, 0.0, 0.0],
                "multi_match_rate": 0.0,
                "success": 0,
                "steps": 0,
                "path_len": 0.0,
                "time":0,
                "distance":0,
                "velocity":0,
                "trans_times":0,
            },
            "current": {
                "mode": init_mode_mask,
                "match_rate": [0.0, 0.0, 0.0, 0.0],
                "multi_match_rate": 0.0,
                "success": 0,
                "steps": 0,
                "path_len": 0.0,
                "time":0,
                "distance":0,
                "velocity":0,
                "trans_times":0,
            }
        }

        return self.state

    def step(self, action):
        self.step_cnt += 1
        reward = 0.0

        self.state["previous"] = dict(self.state["current"])

        time = float(self.current_row['time'].iat[0])
        distance = float(self.current_row['distance'].iat[0])
        velocity = float(self.current_row['velocity'].iat[0])

        prev_mask = np.asarray(self.state["current"]["mode"], dtype=np.int64)
        if int(prev_mask.sum()) == 0:
            prev_mask[np.random.randint(0, 4)] = 1

        cur_mask = self._flip_one_bit(prev_mask, action)
        selected_modes = self._mask_to_modes(cur_mask)

        changed = (cur_mask.tolist() != prev_mask.tolist())
        if changed:
            self.no_change_streak = 0
            reward -= 1
        else:
            self.no_change_streak += 1
        
        match_rate, multi_match_rate, success, steps, path_len, trans_times = self._run_PathMode(selected_modes)

        reward += 0.2 * success # 0~0.2

        reward += multi_match_rate if multi_match_rate >=0.6 else -1

        # 如果选择的mode在match rate中为0，则扣分，否则+match rate
        for i in range(len(cur_mask)):
            if cur_mask[i] == 1 and match_rate[i] == 0:
                reward -= 1

        reward += max(match_rate)


        reward -= min(0.5 * cur_mask.sum(), 5) # 0~0.4

        reward -= min(0.5 * trans_times, 5) # 0~

        # 速度奖励：基于当前 mode 的速度分布偏离程度计算
        reward += self._speed_deviation_reward(cur_mask, velocity)

        self.state["current"] = {
            "mode": cur_mask.tolist(),
            "match_rate": match_rate,
            "multi_match_rate": multi_match_rate,
            "success": int(success),
            "steps": int(steps),
            "path_len": float(path_len),
            "time": time,
            "distance":distance,
            "velocity":velocity,
            "trans_times":trans_times,
        }

        if self.no_change_streak >= self.no_change_patience:
            done = True
            self.finish = True
        elif self.step_cnt >= self.max_mode_steps:
            done = True
            reward -= 100
        else:
            done = False
            
        success = int(success)
        '''        
        print(f"traj NO. = {self.traj_cnt}, "
              f"step = {self.step_cnt}, "
              f"reward = {reward:.3f}, "
              f"selelcted modes = {selected_modes}, "
              f"success = {success}, "
              f"match rate = {match_rate}, "
              f"multi match rate = {multi_match_rate:.3f}, "
              )
        '''

        return self.state, float(reward), done, success, multi_match_rate


if __name__ == "__main__":

    TEST_ENV = 'Mode'  # 'Path' or 'Mode'
    # 测试环境是否配置成功
    if TEST_ENV == 'Path':
        with open('data/GridModesAdjacentRealworld.pkl', 'rb') as f:
            mapdata = pickle.load(f)
        traj = pd.read_csv('data\data_lower_train_random.csv')

        pathenv = PathEnv(train_mode=True, mapdata=mapdata, traj=traj, FOV=5, distance_threshold=1.0)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"env_test_log_{timestamp}.txt"

        with open(log_filename, 'w', encoding='utf-8') as log_file:
            log_file.write("开始测试环境...\n")
            
            for episode in range(5):
                log_file.write(f"\n=================================== Episode {episode + 1} ==========================================\n")
                state = pathenv.reset()
                log_file.write(f"初始状态 - {state}\n")
                log_file.write(f"初始距离: {pathenv.state['total_distance']}\n")
                log_file.write(f"初始位置 - ({pathenv.locx_start}, {pathenv.locy_start}), end: ({pathenv.locx_end}, {pathenv.locy_end})\n")
                log_file.write(f"  初始节点记忆: {pathenv.node_memory}\n")
                log_file.write(f"初始neighbor - {pathenv.neighbor}, end: {pathenv.neighbor}\n")
                log_file.write(f"====================================================================\n")
                step = 0
                done = False
                total_reward = 0

                while not done:
                    # 随机选择动作
                    action = np.random.randint(len(dxdy_dict))
                    
                    # 执行动作前的位置
                    prev_pos = (pathenv.locx_start, pathenv.locy_start)
                    
                    # 执行动作
                    state, r, done, succ = pathenv.step(action)
                    
                    total_reward += r

                    step += 1
                    
                    current_distance = np.abs(pathenv.state['remaining_distance'][0] + pathenv.state['total_distance'][0]) + np.abs(pathenv.state['remaining_distance'][1] + pathenv.state['total_distance'][1])
                    prev_distance = np.abs(pathenv.state['previous_remaining_distance'][0] + pathenv.state['total_distance'][0]) + np.abs(pathenv.state['previous_remaining_distance'][1] + pathenv.state['total_distance'][1])

                    log_file.write(f"Step {step}:\n")
                    log_file.write(f"  最大步数: {pathenv.max_step:.2f}\n")
                    log_file.write(f"  动作: {action}\n")
                    log_file.write(f"  坐标变换: {dxdy_dict[action]}\n")
                    log_file.write(f"  位置变化:  {prev_pos}->{(pathenv.locx_start, pathenv.locy_start)}\n")
                    log_file.write(f"  前一步距离: {prev_distance}\n")
                    log_file.write(f"  距离: {current_distance}\n")
                    log_file.write(f"  当前neighbor: {state['patch']}\n")
                    log_file.write(f"  奖励: {r:.2f}\n")
                    log_file.write(f"  状态: {pathenv.state}\n")
                    log_file.write(f"  节点记忆: {pathenv.node_memory}\n")
                    log_file.write(f"====================================================================\n")
                    
                    if done:
                        log_file.write(f"Episode结束! 原因: {'距离达到阈值' if current_distance <= pathenv.distance_threshold else '达到最大步数'}\n")
                        break
                
                log_file.write(f"\nEpisode {episode + 1} 总结:\n")
                log_file.write(f"  总步数: {step}\n")
                log_file.write(f"  agent总奖励: {total_reward:.2f}\n")
                log_file.write(f"  最终距离: {current_distance:.2f}\n")
    
    elif TEST_ENV == 'Mode':
        with open('data/GridModesAdjacentRealworld.pkl', 'rb') as f:
            mapdata = pickle.load(f)
        traj = pd.read_csv('data\data_lower_train_ordered.csv')
        
        modeenv = ModeEnv(model_path='PathModel\PathModel.pth',
                          mapdata=mapdata, 
                          traj=traj, 
                          train_mode=True,
                          )

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"env_test_log_{timestamp}.txt"

        with open(log_filename, 'w', encoding='utf-8') as log_file:
            log_file.write("开始测试环境...\n")
            
            for episode in range(10):
                log_file.write(f"\n=================================== Episode {episode + 1} ==========================================\n")
                state = modeenv.reset()
                log_file.write(f"初始previous: {modeenv.state['previous']}\n")
                log_file.write(f"初始current: {modeenv.state['current']}\n")
                log_file.write(f"====================================================================\n")
                step = 0
                done = False
                total_reward = 0

                while not done:
                    # 随机选择动作
                    action = np.random.randint(0, 5)
                    log_file.write(f"action: {action}\n")
                                   
                    # 执行动作
                    state, r, done, succ, _ = modeenv.step(action)
                    
                    total_reward += r

                    step += 1

                    log_file.write(f"previous: {modeenv.state['previous']}\n")
                    log_file.write(f"current: {modeenv.state['current']}\n")
                    log_file.write(f"reward: {r}\n")
                    log_file.write(f"====================================================================\n")
                    
                    if done:
                        log_file.write(f"Episode结束!\n")
                        break
                
                log_file.write(f"\nEpisode {episode + 1} 总结:\n")
                log_file.write(f"  总步数: {step}\n")
                log_file.write(f"  agent总奖励: {total_reward:.2f}\n")

    else:
        print("无效的测试环境配置，请选择 'Path' 或 'Mode'。")