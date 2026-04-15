import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200000
    start_steps: int = 2000
    update_after: int = 1000
    update_every: int = 1
    hidden_dim: int = 256
    target_entropy_ratio: float = 0.85
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DiscreteSACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = action_dim

        # actor: 输出每个离散动作的 logits
        self.actor = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)

        # twin Q
        self.q1 = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2 = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2_target = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)

        # 自动温度
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # 离散动作目标熵：接近均匀分布熵
        self.target_entropy = cfg.target_entropy_ratio * np.log(action_dim)

        self.replay = ReplayBuffer(cfg.buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, evaluate: bool = False) -> int:
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(s)
        probs = F.softmax(logits, dim=-1)

        if evaluate:
            a = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample().item()
        return int(a)

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ===== Q target =====
        with torch.no_grad():
            next_logits = self.actor(ns)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()

            q1_t = self.q1_target(ns)
            q2_t = self.q2_target(ns)
            q_t_min = torch.min(q1_t, q2_t)

            # V(s') = sum pi(a|s')[Q - alpha*logpi]
            next_v = (next_probs * (q_t_min - self.alpha.detach() * next_log_probs)).sum(dim=-1, keepdim=True)
            q_target = r + (1.0 - d) * self.cfg.gamma * next_v

        q1_all = self.q1(s)
        q2_all = self.q2(s)
        q1_sa = q1_all.gather(1, a)
        q2_sa = q2_all.gather(1, a)

        q1_loss = F.mse_loss(q1_sa, q_target)
        q2_loss = F.mse_loss(q2_sa, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # ===== Actor =====
        logits = self.actor(s)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        with torch.no_grad():
            q_min = torch.min(self.q1(s), self.q2(s))

        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ===== Alpha =====
        entropy = -(probs * log_probs).sum(dim=-1)  # H(pi)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ===== soft update =====
        self._soft_update(self.q1, self.q1_target, self.cfg.tau)
        self._soft_update(self.q2, self.q2_target, self.cfg.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    @staticmethod
    def _soft_update(net, target_net, tau):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)