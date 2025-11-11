# -*- coding: utf-8 -*-
# Hybrid POMDP RL (PF-DRQN)
# - Torch PF + tensor-direct I/O + fixed replay sampler
# - FIX: torch.compile() 래핑으로 생기는 state_dict 키 mismatch 해결(safe_state_dict, 초기화 순서/동기화 변경)
# - torch.compile: Triton 없으면 aot_eager로 자동 폴백 (Windows/로컬 안전)
# - Opponent uses epsilon-greedy (ε=0.05) during training phases
# - Step guard 1,000 + "no-progress" early stop
# - Every 1,000 episodes per phase: save logs + overwrite applied policy (.pt + meta.json)
# - From evaluation pool, take top-10% episodes -> filter "significant" -> write JSON
# - After some outers, also run big-map(40x40) eval and write JSON (test-only)

import os, time, json, math, random, threading
from dataclasses import dataclass
from typing import Tuple, List
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- I/O paths ------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
APPLIED_DIR  = os.path.join(OUTPUT_DIR, "applied")
SNAP_DIR     = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR     = os.path.join(OUTPUT_DIR, "logs")
EVAL_DIR     = os.path.join(OUTPUT_DIR, "eval")
for d in [APPLIED_DIR, SNAP_DIR, LOGS_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------- Basic utils ----------------------------------- #
def clamp(x, lo, hi): return max(lo, min(hi, x))
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def angle_bin_from_dxdy16(dx, dy):
    if dx == 0 and dy == 0: return 0
    ang = math.atan2(dy, dx)
    if ang < 0: ang += 2*math.pi
    bin_size = 2*math.pi/16
    return int((ang + bin_size/2)//bin_size) % 16
DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def dir_to_vec(di): return DIRS8[di % 8]
def add_pos(p, delta, n): return (max(0, min(n-1, p[0]+delta[0])), max(0, min(n-1, p[1]+delta[1])))

# --------------------------- Configs --------------------------------------- #
@dataclass
class EnvConfig:
    grid_size: int = 20
    fov_radius: int = 10
    obs_noise: float = 0.12
    slip: float = 0.04
    capture_radius: int = 0
    seed: int = 42

    # reward shaping
    time_penalty: float = 0.0005
    k_distance: float  = 0.005
    k_los: float       = 0.002
    k_info: float      = 0.0005

    # action costs
    action_cost_rot: float      = 0.0003
    action_cost_look: float     = 0.00035
    action_cost_focus: float    = 0.00045
    action_cost_strafe: float   = 0.00015
    action_cost_back: float     = 0.00015
    action_cost_spin: float     = 0.0003
    action_cost_take_corner: float = 0.0005
    action_cost_collect: float  = 0.0003
    action_cost_decoy: float    = 0.0008

    # coins
    coin_count: int = 5
    coin_collect_ticks: int = 5
    r_collect_tick: float = 0.3
    r_collect_done: float = 1.2
    r_all_coins_bonus: float = 6.0

    # coin shaping
    r_coin_detect: float = 0.07
    k_coin_approach: float = 0.003
    r_coin_visible: float = 0.015
    r_duplicate_penalty: float = -0.06

    # evader threat shaping
    threat_radius: int = 3
    k_threat_avoid: float = 0.003

    # decoy
    decoy_duration: int = 3
    r_decoy_success: float = 0.5
    decoy_slip_add: float = 0.08
    decoy_noise_mult: float = 2.0

# Hybrid model + PF config
@dataclass
class HybridConfig:
    # PF
    pf_num_particles: int = 200
    pf_resample_every: int = 3
    pf_ess_threshold: float = 0.5
    pf_process_noise: float = 0.7
    pf_lik_ang_sigma: float = 1.0
    pf_lik_dist_sigma: float = 1.0
    pf_topk_modes: int = 3

    # obs enc
    obs_dim_onehot: int = 41                # ang16 + dist10 + far1 + face8 + coin4 + see2
    belief_feat_dim: int = 11               # mean2 + meanDist1 + varTrace1 + entropy1 + topk*2(=6)

    # DRQN
    hidden: int = 128
    n_actions: int = 15
    gamma: float = 0.997
    lr: float = 3e-4
    n_step: int = 1

    burn_in: int = 12
    unroll: int = 24
    target_update: int = 2000

    replay_capacity: int = 100_000
    batch_size: int = 96
    seq_len: int = 36

    eps_start: float = 0.20
    eps_final: float = 0.02
    eps_decay: float = 0.9995

    # opponent epsilon
    opponent_eps: float = 0.05

# trainer schedule
@dataclass
class TrainSchedule:
    SHOW_EVERY: int = 1000
    PRINT_EVERY: int = 50
    MAX_STEPS: int = 1000
    EVAL_EVERY: int = 100
    POOL_LIMIT: int = 300
    OUTER_SWITCH_BIGMAP: int = 5

# --------------------------- Actions (15) ----------------------------------- #
ACTIONS: List[Tuple[str, Tuple]] = [
    ("WAIT", ()),
    ("ROT+1", (+1,)), ("ROT-1", (-1,)),
    ("ROT+2", (+2,)), ("ROT-2", (-2,)),
    ("SPIN", ()),
    ("FORWARD", (1,)),
    ("STRAFE_L", ("L", 1)), ("STRAFE_R", ("R", 1)),
    ("BACKSTEP", (1,)),
    ("LOOK", ()),
    ("FOCUS", ()),
    ("TAKE_CORNER", ()),
    ("COLLECT", ()),
    ("DECOY", ()),
]
NUM_ACTIONS = len(ACTIONS)

# --------------------------- Environment (1v1) ------------------------------ #
class TwoAgentTagEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser": 0, "evader": 0}
        self.coins = set(); self.coin_ticks = {}
        self.evader_collected = 0
        self.coins_seen = set()
        self.decoy_timer = 0
        self.state = None

    def reset(self):
        while True:
            c = (random.randrange(self.n), random.randrange(self.n))
            e = (random.randrange(self.n), random.randrange(self.n))
            if manhattan(c,e) >= self.n//2: break
        self.state = (c, 0, e, 4)
        self.look_bonus = {"chaser": 0, "evader": 0}
        self.decoy_timer = 0
        self.evader_collected = 0
        self.coins_seen.clear()

        self.coins.clear(); self.coin_ticks.clear()
        forbid = {self.state[0], self.state[2]}
        while len(self.coins) < self.cfg.coin_count:
            p = (random.randrange(self.n), random.randrange(self.n))
            if p not in forbid and p not in self.coins:
                self.coins.add(p); self.coin_ticks[p] = 0
        return self.state

    def _delta_face_to(self, face_idx, dx, dy):
        desired = angle_bin_from_dxdy16(dx, dy) % 8
        diff = (desired - face_idx) % 8
        if diff == 0: return 0
        return +1 if diff <= 4 else -1

    def _apply_action(self, pos, face_idx, who, a_idx, other_pos):
        kind, param = ACTIONS[a_idx]
        n = self.n; new_pos, new_face = pos, face_idx

        if kind == "WAIT":
            pass
        elif kind.startswith("ROT"):
            step = param[0]; new_face = (new_face + step) % 8
        elif kind == "SPIN":
            new_face = (new_face + 4) % 8
        elif kind == "FORWARD":
            step = param[0]; fdx,fdy = dir_to_vec(new_face)
            for _ in range(step): new_pos = add_pos(new_pos, (fdx,fdy), n)
        elif kind == "STRAFE_L":
            sdx,sdy = dir_to_vec(new_face - 2)
            new_pos = add_pos(new_pos, (sdx,sdy), n)
        elif kind == "STRAFE_R":
            sdx,sdy = dir_to_vec(new_face + 2)
            new_pos = add_pos(new_pos, (sdx,sdy), n)
        elif kind == "BACKSTEP":
            bdx,bdy = dir_to_vec(new_face + 4)
            new_pos = add_pos(new_pos, (bdx,bdy), n)
        elif kind == "LOOK":
            self.look_bonus["chaser" if who=="chaser" else "evader"] = max(self.look_bonus["chaser" if who=="chaser" else "evader"], 1)
        elif kind == "FOCUS":
            turns = max(2, self.look_bonus["chaser" if who=="chaser" else "evader"])
            self.look_bonus["chaser" if who=="chaser" else "evader"] = turns
        elif kind == "TAKE_CORNER":
            dx, dy = other_pos[0]-pos[0], other_pos[1]-pos[1]
            step = self._delta_face_to(new_face, dx, dy)
            new_face = (new_face + step) % 8
            fdx,fdy = dir_to_vec(new_face)
            new_pos = add_pos(new_pos, (fdx,fdy), n)
        elif kind in ("COLLECT","DECOY"):
            pass
        return new_pos, new_face

    def _nearest_coin_dist(self, epos):
        if not self.coins: return None, 99
        dists = [(p, manhattan(epos,p)) for p in self.coins]
        p,d = min(dists, key=lambda x:x[1])
        return p,d

    def step(self, a_c, a_e, eval_mode=False):
        (c, fc, e, fe) = self.state
        prev_d = manhattan(c,e)

        if self.look_bonus["chaser"]>0: self.look_bonus["chaser"]-=1
        if self.look_bonus["evader"]>0: self.look_bonus["evader"]-=1
        if self.decoy_timer>0: self.decoy_timer-=1

        slip_c = self.cfg.slip + (self.cfg.decoy_slip_add if self.decoy_timer>0 else 0.0)
        slip_e = self.cfg.slip
        if not eval_mode:
            if random.random() < slip_c: a_c = random.randrange(NUM_ACTIONS)
            if random.random() < slip_e: a_e = random.randrange(NUM_ACTIONS)

        c_next, fc_next = self._apply_action(c, fc, "chaser", a_c, e)
        e_next, fe_next = self._apply_action(e, fe, "evader", a_e, c)

        # decoy
        r_decoy = 0.0
        if ACTIONS[a_e][0] == "DECOY":
            if manhattan(e_next, c_next) <= self.cfg.fov_radius:
                self.decoy_timer = self.cfg.decoy_duration
                r_decoy = self.cfg.r_decoy_success

        # collect
        r_collect_tick = 0.0; r_collect_done = 0.0; dup_pen = 0.0
        if ACTIONS[a_e][0] == "COLLECT":
            if e_next in self.coins:
                self.coin_ticks[e_next] += 1
                r_collect_tick = self.cfg.r_collect_tick
                if self.coin_ticks[e_next] >= self.cfg.coin_collect_ticks:
                    self.coins.remove(e_next); del self.coin_ticks[e_next]
                    self.evader_collected += 1
                    r_collect_done = self.cfg.r_collect_done
            else:
                dup_pen = self.cfg.r_duplicate_penalty

        self.state = (c_next, fc_next, e_next, fe_next)

        done_capture  = (manhattan(c_next,e_next) <= self.cfg.capture_radius)
        done_allcoins = (self.evader_collected >= self.cfg.coin_count)
        done = done_capture or done_allcoins

        r_c = 1.0 if done_capture else 0.0
        r_e = (-1.0 if done_capture else 0.02)

        # coin shaping
        nearest_p, d_now = self._nearest_coin_dist(e_next)
        if nearest_p is not None and manhattan(e_next, nearest_p) <= self.cfg.fov_radius:
            if nearest_p not in self.coins_seen:
                r_e += self.cfg.r_coin_detect; self.coins_seen.add(nearest_p)
            r_e += self.cfg.r_coin_visible
        _, d_prev = self._nearest_coin_dist(e)
        if d_prev!=99 and d_now!=99:
            r_e += self.cfg.k_coin_approach * (d_prev - d_now)
        r_e += r_collect_tick + r_collect_done + r_decoy + dup_pen
        if done_allcoins: r_e += self.cfg.r_all_coins_bonus

        # generic shaping
        r_c -= self.cfg.time_penalty; r_e -= self.cfg.time_penalty
        new_d  = manhattan(c_next, e_next)
        delta_d = prev_d - new_d
        r_c += self.cfg.k_distance * (delta_d)
        r_e += self.cfg.k_distance * (-delta_d)
        if new_d <= self.cfg.fov_radius: r_c += self.cfg.k_los
        else:                             r_e += self.cfg.k_los

        kind_c,_ = ACTIONS[a_c]; kind_e,_ = ACTIONS[a_e]
        if kind_c == "LOOK":  r_c += self.cfg.k_info
        if kind_c == "FOCUS": r_c += 0.75*self.cfg.k_info
        if kind_e == "LOOK":  r_e += self.cfg.k_info
        if kind_e == "FOCUS": r_e += 0.75*self.cfg.k_info

        if new_d <= self.cfg.threat_radius and kind_e not in ("COLLECT","DECOY"):
            r_e += self.cfg.k_threat_avoid

        # action costs
        def cost(kind):
            if   kind.startswith("ROT"): return self.cfg.action_cost_rot
            elif kind == "LOOK":        return self.cfg.action_cost_look
            elif kind == "FOCUS":       return self.cfg.action_cost_focus
            elif kind.startswith("STRAFE"): return self.cfg.action_cost_strafe
            elif kind == "BACKSTEP":    return self.cfg.action_cost_back
            elif kind == "SPIN":        return self.cfg.action_cost_spin
            elif kind == "TAKE_CORNER": return self.cfg.action_cost_take_corner
            elif kind == "COLLECT":     return self.cfg.action_cost_collect
            elif kind == "DECOY":       return self.cfg.action_cost_decoy
            else: return 0.0
        r_c -= cost(kind_c); r_e -= cost(kind_e)

        if done_capture: r_c = 1.0; r_e = -1.0
        return self.state, (r_c, r_e), done

    # obs
    def _dist_bin(self, d, cuts=(1,2,4,6,9,13,18,24)):
        for i,c in enumerate(cuts):
            if d<=c: return i
        return len(cuts)
    def _coin_dist_bin(self, d):
        if d<=2: return 0
        if d<=5: return 1
        if d<=9: return 2
        return 3

    def observe(self, for_chaser: bool, eval_mode=False):
        (c,fc,e,fe) = self.state
        me_pos, me_face = (c,fc) if for_chaser else (e,fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"

        bonus_turns = self.look_bonus[key]
        extra_noise_mult = (self.cfg.decoy_noise_mult if (for_chaser and self.decoy_timer>0) else 1.0)

        if not eval_mode and bonus_turns >= 2:
            eff_fov = int(self.cfg.fov_radius * 1.25); base_noise = self.cfg.obs_noise * 0.5
        elif not eval_mode and bonus_turns >= 1:
            eff_fov = int(self.cfg.fov_radius * 1.5);  base_noise = self.cfg.obs_noise * 0.5
        else:
            eff_fov = self.cfg.fov_radius;            base_noise = self.cfg.obs_noise
        eff_noise = 0.0 if eval_mode else base_noise * extra_noise_mult

        dx,dy = ot_pos[0]-me_pos[0], ot_pos[1]-me_pos[1]
        d = abs(dx)+abs(dy); far = d > eff_fov
        if far:
            ang_bin = 0; dist_bin = 9
        else:
            ang_bin = angle_bin_from_dxdy16(dx,dy)
            dist_bin = self._dist_bin(d)
            if random.random()<eff_noise: ang_bin = (ang_bin + random.choice([-1,1,2]))%16
            if random.random()<eff_noise: dist_bin = max(0, min(8, dist_bin+random.choice([-1,1])))
        my_face_bin = me_face

        nearest_p, nearest_d = self._nearest_coin_dist(e)
        coin_dist_bin = 3; see_coin = 0
        if nearest_p is not None:
            coin_dist_bin = self._coin_dist_bin(nearest_d)
            if nearest_d <= eff_fov:
                see_coin = 1
                if not eval_mode and random.random()<eff_noise:
                    see_coin = 1 - see_coin
        return (ang_bin, dist_bin, d>eff_fov, my_face_bin, coin_dist_bin, see_coin)

# --------------------------- Representation -------------------------------- #
class BeliefIndexer:
    def __init__(self, na=16, nd=10, nf=8, nc=4, ns=2):
        self.na, self.nd, self.nf, self.nc, self.ns = na, nd, nf, nc, ns
    def index(self, obs):
        ang, dist, far, face, cbin, see = obs
        dcode = 9 if far else dist
        return (((((face*self.nd)+dcode)*self.na + (ang%self.na))*self.nc + cbin)*self.ns + see)
    @property
    def n_bins(self): return self.na*self.nd*self.nf*self.nc*self.ns

def obs_to_onehot_torch(obs, device):
    ang, dist, far, face, coin_bin, see = obs
    v=[]
    def oh(i,n):
        t = torch.zeros(n, device=device, dtype=torch.float32)
        t[int(i)%n] = 1.0
        return t
    v.append(oh(ang,16))
    v.append(oh(9 if far else dist,10))
    v.append(torch.tensor([1.0 if far else 0.0], device=device, dtype=torch.float32))
    v.append(oh(face,8))
    v.append(oh(coin_bin,4))
    v.append(oh(see,2))
    return torch.cat(v, dim=0)  # (41,)

# --------------------------- Torch Particle Filter -------------------------- #
class TorchParticleFilter:
    def __init__(self, hycfg: HybridConfig, grid_size:int, fov_radius:int, seed:int=0, device:str="cpu"):
        self.cfg = hycfg
        self.n = grid_size; self.fov = fov_radius
        self.num = hycfg.pf_num_particles
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self):
        low = -self.n + 1; high = self.n - 1
        self.p = torch.randint(low, high + 1, (self.num, 2), generator=self.rng,
                               device=self.device, dtype=torch.float32)
        self.w = torch.full((self.num,), 1.0/self.num, device=self.device, dtype=torch.float32)
        self._steps = 0

    def predict(self, my_move=(0,0)):
        dx, dy = my_move
        self.p[:,0] -= dx; self.p[:,1] -= dy
        jitter = torch.normal(mean=0.0, std=self.cfg.pf_process_noise,
                              size=self.p.shape, generator=self.rng, device=self.device)
        self.p.add_(jitter)
        self.p[:,0].clamp_(-self.n+1, self.n-1)
        self.p[:,1].clamp_(-self.n+1, self.n-1)
        self._steps += 1

    def weight_update(self, obs):
        ang, dist, far, *_ = obs
        ang_t = torch.as_tensor(ang, device=self.device, dtype=torch.int64)
        dist_code = torch.as_tensor(9 if far else dist, device=self.device, dtype=torch.int64)

        dx = self.p[:,0]; dy = self.p[:,1]
        dist_mh = dx.abs() + dy.abs()
        pred_far = dist_mh > self.fov

        ang_raw = torch.atan2(dy, dx)
        ang_raw = torch.where(ang_raw < 0, ang_raw + 2*math.pi, ang_raw)
        bin_size = 2*math.pi / 16
        pred_ang = torch.remainder(((ang_raw + bin_size/2) // bin_size), 16).to(torch.int64)
        pred_ang = torch.where(pred_far, torch.zeros_like(pred_ang), pred_ang)

        cuts = torch.tensor([1,2,4,6,9,13,18,24], device=self.device, dtype=torch.float32)
        pred_dist = torch.searchsorted(cuts, dist_mh, right=True).clamp(max=9).to(torch.int64)
        pred_dist = torch.where(pred_far, torch.full_like(pred_dist, 9), pred_dist)

        ang_err = (pred_ang - ang_t).remainder(16).abs()
        ang_err = torch.minimum(ang_err, 16 - ang_err).to(torch.float32)
        dist_err = (pred_dist - dist_code).abs().to(torch.float32)

        like_ang = torch.exp(-(ang_err**2)/(2*(self.cfg.pf_lik_ang_sigma**2)))
        like_dst = torch.exp(-(dist_err**2)/(2*(self.cfg.pf_lik_dist_sigma**2)))
        like = like_ang * like_dst + 1e-8

        self.w.mul_((like))
        s = self.w.sum()
        if torch.isfinite(s) and s > 0:
            self.w.div_(s)
        else:
            self.w.fill_((1.0/self.num))

        ess = 1.0 / torch.sum(self.w**2)
        if (ess/self.num) < self.cfg.pf_ess_threshold or (self._steps % self.cfg.pf_resample_every) == 0:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.num
        u = torch.rand((), generator=self.rng, device=self.device, dtype=torch.float64)
        positions = (torch.arange(N, device=self.device, dtype=torch.float64) + u) / N
        cumsum = torch.cumsum(self.w.to(torch.float64), dim=0)
        cumsum[-1] = 1.0
        idx = torch.searchsorted(cumsum, positions).clamp(max=N-1).to(torch.long)
        self.p = self.p.index_select(0, idx)
        self.w.fill_(1.0/N)

    def summarize(self, topk=None):
        if topk is None: topk = self.cfg.pf_topk_modes
        w = self.w.clamp_min(1e-12)
        mean = (w[:, None] * self.p).sum(dim=0)
        var_trace = (w[:, None] * (self.p - mean).pow(2)).sum(dim=0).sum()
        mean_dist = (self.p.abs().sum(dim=1) * w).sum()
        entropy = float((-(w * torch.log(w))).sum() / math.log(len(w)))

        k = min(topk, self.num)
        top_idx = torch.topk(w, k=k, largest=True).indices
        top = self.p.index_select(0, top_idx)

        s = float(self.n)
        feats = [mean[0]/s, mean[1]/s, mean_dist/(2*s), min(1.0, float(var_trace/(s*s))), entropy]
        for i in range(topk):
            if i < top.shape[0]:
                feats.extend([float(top[i, 0]/s), float(top[i, 1]/s)])
            else:
                feats.extend([0.0, 0.0])
        return torch.tensor(feats, device=self.device, dtype=torch.float32)

# --------------------------- DRQN model ------------------------------------ #
class DRQN(nn.Module):
    def __init__(self, in_dim, hidden, n_actions):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU())
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.val = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Linear(128,1))
        self.adv = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Linear(128,n_actions))
    def forward(self, x_seq, h0=None):
        z = self.enc(x_seq)
        out, hT = self.gru(z, h0)
        V = self.val(out); A = self.adv(out)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hT

# --------------------------- Compile-safe state_dict ------------------------ #
def safe_state_dict(module: torch.nn.Module):
    """torch.compile 래핑(_orig_mod) 여부와 무관하게 순정 키로 state_dict 반환"""
    return getattr(module, "_orig_mod", module).state_dict()

# --------------------------- Replay buffer --------------------------------- #
class SeqReplay:
    def __init__(self, capacity:int): self.capacity=capacity; self.data=[]; self.ptr=0
    def push_episode(self, traj):
        if len(self.data)<self.capacity: self.data.append(traj)
        else: self.data[self.ptr]=traj; self.ptr=(self.ptr+1)%self.capacity
    def __len__(self): return len(self.data)
    # FIX: 짧은 에피소드 때문에 전체 배치를 포기하지 않음
    def sample_batch(self, batch:int, burn:int, unroll:int):
        if len(self.data) < batch:
            return None
        need = burn + unroll + 1
        candidates = [ep for ep in self.data if len(ep["act"]) >= need]
        if len(candidates) < batch:
            return None
        seqs = random.sample(candidates, batch)
        out=[]
        for ep in seqs:
            T=len(ep["act"])
            s=random.randint(0, T-need)
            sl=slice(s,s+need)
            out.append({k:(v[sl] if isinstance(v, np.ndarray) else v) for k,v in ep.items()})
        return out

# --------------------------- Agent ----------------------------------------- #
class HybridAgent:
    def __init__(self, hy: HybridConfig, device="cpu"):
        self.cfg=hy
        in_dim = hy.obs_dim_onehot + hy.belief_feat_dim

        base = DRQN(in_dim, hy.hidden, hy.n_actions).to(device)   # compile 전

        # 타깃: 컴파일 전 state_dict로 동기화(키 동일)
        self.tgt = DRQN(in_dim, hy.hidden, hy.n_actions).to(device)
        self.tgt.load_state_dict(base.state_dict())

        # compile 시도(없으면 aot_eager 폴백)
        self.net = self._maybe_compile(base, device)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=hy.lr)
        self.device=device
        self.eps=hy.eps_start
        self.step_count=0
        self._scaler = torch.amp.GradScaler('cuda') if device.startswith("cuda") else None

    def _maybe_compile(self, base, device):
        if not device.startswith("cuda"):
            return base
        try:
            import triton  # 존재 확인
            return torch.compile(base, mode="reduce-overhead")
        except Exception:
            # Triton 미설치/문제 시 aot_eager로 폴백(안전)
            try:
                return torch.compile(base, mode="reduce-overhead", backend="aot_eager")
            except Exception:
                return base  # 최종 안전 폴백

    def decay_eps(self): self.eps = max(self.cfg.eps_final, self.eps*self.cfg.eps_decay)

    # tensor-direct path
    def q_eval_t(self, x_t: torch.Tensor, h=None):
        xt = x_t.view(1,1,-1)
        with torch.no_grad():
            q,h2 = self.net(xt, h)
        return q, h2

    def act_eps_t(self, x_t: torch.Tensor, h=None):
        q,h2 = self.q_eval_t(x_t, h)
        if random.random() < self.eps: a = random.randrange(q.shape[-1])
        else: a = int(torch.argmax(q, dim=-1).item())
        return a, h2

    def act_eps_fixed_t(self, x_t: torch.Tensor, eps: float, h=None):
        q,h2 = self.q_eval_t(x_t, h)
        if random.random() < eps: a = random.randrange(q.shape[-1])
        else: a = int(torch.argmax(q, dim=-1).item())
        return a, h2

    def act_greedy_t(self, x_t: torch.Tensor, h=None):
        q,h2=self.q_eval_t(x_t,h)
        return int(torch.argmax(q, dim=-1).item()), h2

    def train_batch(self, batch, gamma, burn, unroll):
        if batch is None: return 0.0
        B=len(batch); dev=self.device
        def to_t(name):
            arr = np.stack([b[name] for b in batch], 0)
            return torch.from_numpy(arr).to(dev, dtype=torch.float32, non_blocking=True)
        x = to_t("x")
        a = torch.from_numpy(np.stack([b["act"] for b in batch],0)).long().to(dev, non_blocking=True)
        r = to_t("rew"); d = to_t("done")

        with torch.no_grad():
            _, h = self.net(x[:, :burn, :])

        use_amp = (self._scaler is not None)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                q_on, _   = self.net(x[:, burn:-1, :], h)
                q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1)
                q_on_next,_  = self.net(x[:, burn+1:, :], h)
                a_star = torch.argmax(q_on_next, dim=-1)
                q_tgt_next,_ = self.tgt(x[:, burn+1:, :], h)
                q_next = q_tgt_next.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
                target = r[:, burn:burn+unroll] + gamma*(1.0 - d[:, burn:burn+unroll])*q_next
                loss = F.smooth_l1_loss(q_sel, target)
            self.optim.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self._scaler.step(self.optim); self._scaler.update()
        else:
            q_on, _   = self.net(x[:, burn:-1, :], h)
            q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                q_on_next,_  = self.net(x[:, burn+1:, :], h)
                a_star = torch.argmax(q_on_next, dim=-1)
                q_tgt_next,_ = self.tgt(x[:, burn+1:, :], h)
                q_next = q_tgt_next.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
            target = r[:, burn:burn+unroll] + gamma*(1.0 - d[:, burn:burn+unroll])*q_next
            loss = F.smooth_l1_loss(q_sel, target)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optim.step()

        self.step_count += 1
        if (self.step_count % self.cfg.target_update)==0:
            # compile 여부와 무관하게 안전하게 동기화
            self.tgt.load_state_dict(safe_state_dict(self.net))
        return float(loss.item())

# --------------------------- Helpers --------------------------------------- #
def my_move_delta_from_action(a_idx:int, my_face_bin:int):
    if a_idx==6:   vx,vy = dir_to_vec(my_face_bin)
    elif a_idx==7: vx,vy = dir_to_vec((my_face_bin-2)%8)
    elif a_idx==8: vx,vy = dir_to_vec((my_face_bin+2)%8)
    elif a_idx==9: vx,vy = dir_to_vec((my_face_bin+4)%8)
    else:          vx,vy = (0,0)
    return (vx,vy)

def rle_encode(seq):
    if not seq: return []
    out=[]; cur=seq[0]; cnt=1
    for x in seq[1:]:
        if x==cur: cnt+=1
        else: out.append([int(cur), int(cnt)]); cur=x; cnt=1
    out.append([int(cur), int(cnt)])
    return out

def significant_episode(ep):
    steps = ep["steps"]
    unique = len(set(ep["traj_bins"]))
    act_hist = Counter(ep.get("actions",[]))
    top_act_ratio = (act_hist.most_common(1)[0][1]/max(1, len(ep.get("actions",[])))) if act_hist else 1.0
    return (steps >= 30) and (unique >= 10) and (top_act_ratio <= 0.95)

# --------------------------- Play + Learn ---------------------------------- #
def play_episode_and_learn(env: TwoAgentTagEnv, who: str,
                           agent_self: HybridAgent, agent_other: HybridAgent,
                           pf_self: TorchParticleFilter, pf_oth: TorchParticleFilter,
                           replay: SeqReplay,
                           schedule: TrainSchedule,
                           hy: HybridConfig,
                           no_progress_limit:int=80):
    env.reset(); pf_self.reset(); pf_oth.reset()
    xs, acts, rews, dones = [], [], [], []
    h_self=None; h_oth=None
    sum_r_c=0.0; sum_r_e=0.0; captured=False
    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=False)
    ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=False)
    last_b = None; stagnant=0

    for _ in range(schedule.MAX_STEPS):
        x_self_t = torch.cat([obs_to_onehot_torch(ob_s, agent_self.device),
                              pf_self.summarize()], dim=0)
        x_oth_t  = torch.cat([obs_to_onehot_torch(ob_o, agent_other.device),
                              pf_oth.summarize()], dim=0)

        a_s, h_self = agent_self.act_eps_t(x_self_t, h_self)
        a_o, h_oth  = agent_other.act_eps_fixed_t(x_oth_t, hy.opponent_eps, h_oth)

        if who=="chaser":
            _, (r_c, r_e), done = env.step(a_s, a_o, eval_mode=False); r_who = r_c
        else:
            _, (r_c, r_e), done = env.step(a_o, a_s, eval_mode=False); r_who = r_e

        sum_r_c += r_c; sum_r_e += r_e

        my_face_bin_s = ob_s[3]
        pf_self.predict(my_move=my_move_delta_from_action(a_s, my_face_bin_s))
        ob_s_next = env.observe(for_chaser=(who=="chaser"), eval_mode=False)
        pf_self.weight_update(ob_s_next)

        my_face_bin_o = ob_o[3]
        pf_oth.predict(my_move=my_move_delta_from_action(a_o, my_face_bin_o))
        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=False)
        pf_oth.weight_update(ob_o)

        xs.append(x_self_t.detach().cpu().numpy())
        acts.append(a_s); rews.append(r_who); dones.append(1.0 if done else 0.0)

        b_now = BeliefIndexer().index(ob_s_next)
        if last_b is None or b_now != last_b: stagnant=0; last_b=b_now
        else: stagnant += 1
        ob_s = ob_s_next

        if done or stagnant >= no_progress_limit:
            captured = done
            break

    traj = {"x":np.stack(xs,0).astype(np.float32),
            "act":np.array(acts, dtype=np.int64),
            "rew":np.array(rews, dtype=np.float32),
            "done":np.array(dones, dtype=np.float32)}
    replay.push_episode(traj)

    batch = replay.sample_batch(agent_self.cfg.batch_size, agent_self.cfg.burn_in, agent_self.cfg.unroll)
    if batch is not None:
        loss = agent_self.train_batch(batch, agent_self.cfg.gamma, agent_self.cfg.burn_in, agent_self.cfg.unroll)
        agent_self.decay_eps()
    else:
        loss = 0.0
    return len(acts), sum_r_c, sum_r_e, captured, float(loss)

# --------------------------- Eval (greedy) --------------------------------- #
def rollout_greedy(env: TwoAgentTagEnv, who: str,
                   agent_self: HybridAgent, agent_other: HybridAgent,
                   pf_self: TorchParticleFilter,
                   max_steps:int=1000):
    env.reset(); pf_self.reset()
    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
    traj_bins=[BeliefIndexer().index(ob_s)]
    actions=[]; sum_r=0.0; h_s=None; h_o=None
    for _ in range(max_steps):
        x_self_t = torch.cat([obs_to_onehot_torch(ob_s, agent_self.device),
                              pf_self.summarize()], dim=0)
        a_s, h_s = agent_self.act_greedy_t(x_self_t, h_s)

        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
        x_o_t  = torch.cat([obs_to_onehot_torch(ob_o, agent_other.device),
                            pf_self.summarize()], dim=0)
        a_o, h_o = agent_other.act_greedy_t(x_o_t, h_o)

        if who=="chaser":
            _, (r_c,r_e), done = env.step(a_s, a_o, eval_mode=True); sum_r += r_c
        else:
            _, (r_c,r_e), done = env.step(a_o, a_s, eval_mode=True); sum_r += r_e

        my_face = ob_s[3]; pf_self.predict(my_move=my_move_delta_from_action(a_s, my_face))
        ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
        pf_self.weight_update(ob_s)

        actions.append(a_s)
        traj_bins.append(BeliefIndexer().index(ob_s))
        if done: break

    return {
        "sum_reward": float(sum_r),
        "steps": len(traj_bins)-1,
        "traj_bins_rle": rle_encode(traj_bins),
        "actions_hist": dict(Counter(actions)),
        "actions_seq_head": actions[:50],
    }

# --------------------------- Save helpers ---------------------------------- #
def save_applied(role:str, agent:HybridAgent):
    pt  = os.path.join(APPLIED_DIR, f"{role}_policy.pt")
    meta= os.path.join(APPLIED_DIR, f"{role}_policy.meta.json")
    # compile 여부와 무관하게 안전 저장
    torch.save(safe_state_dict(agent.net), pt)
    m = {
        "role": role, "arch":"Dueling-DRQN(GRU)",
        "obs_dim": agent.cfg.obs_dim_onehot, "belief_dim": agent.cfg.belief_feat_dim,
        "hidden": agent.cfg.hidden, "n_actions": agent.cfg.n_actions,
        "gamma": agent.cfg.gamma, "eps": agent.eps, "step_count": agent.step_count,
        "pytorch": torch.__version__, "cuda": torch.version.cuda, "device": agent.device
    }
    with open(meta,"w",encoding="utf-8") as f: json.dump(m, f, ensure_ascii=False, indent=2)

def save_phase_log(role:str, outer:int, stamp:str, stats:dict):
    path = os.path.join(LOGS_DIR, f"log_{role}_outer{outer}_{stamp}.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(stats, f, ensure_ascii=False, indent=2)

def save_pool_json(role:str, outer:int, stamp:str, top_eps:List[dict]):
    path = os.path.join(EVAL_DIR, f"top10p_{role}_outer{outer}_{stamp}.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(top_eps, f, ensure_ascii=False, indent=2)

def save_significant_json(role:str, outer:int, stamp:str, top_eps:List[dict]):
    sig = [ep for ep in top_eps if significant_episode(
        {"steps": sum(cnt for _,cnt in ep["traj_bins_rle"]),
         "traj_bins": [i for i,_ in ep["traj_bins_rle"]],
         "actions": list(ep.get("actions_hist", {}).keys())
    })]
    path = os.path.join(EVAL_DIR, f"significant_top10p_{role}_outer{outer}_{stamp}.json")
    with open(path,"w",encoding="utf-8") as f: json.dump(sig, f, ensure_ascii=False, indent=2)

# --------------------------- Training loop --------------------------------- #
class StopFlag:
    def __init__(self): self.flag=False
def stop_listener(sf:StopFlag):
    try:
        s=input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s=='q': sf.flag=True
    except: pass

def train_forever(env_cfg:EnvConfig=EnvConfig(),
                  hycfg:HybridConfig=HybridConfig(),
                  schedule:TrainSchedule=TrainSchedule()):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"CUDA available: {use_cuda} | torch: {torch.__version__}")
    if use_cuda:
        try:
            # 최신 TF32 설정 API 사용 (경고 제거)
            torch.backends.cuda.matmul.fp32_precision = "high"   # 또는 "ieee"
            torch.backends.cudnn.conv.fp32_precision  = "tf32"
            torch.backends.cudnn.benchmark = True
            print("Device:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("CUDA init note:", e)

    env = TwoAgentTagEnv(env_cfg)
    chaser = HybridAgent(hycfg, device=device)
    evader = HybridAgent(hycfg, device=device)
    pf_ch = TorchParticleFilter(hycfg, env_cfg.grid_size, env_cfg.fov_radius, seed=0, device=device)
    pf_ev = TorchParticleFilter(hycfg, env_cfg.grid_size, env_cfg.fov_radius, seed=1, device=device)
    replay = SeqReplay(hycfg.replay_capacity)

    sf=StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    outer = 1
    while not sf.flag:
        # ---------------- Phase A: Chaser update ---------------- #
        ep_pool=[]; steps_acc=0; rc_acc=0.0; re_acc=0.0; cap_acc=0; avg_loss=0.0; nloss=0
        t0=time.time()
        for ep in range(1, schedule.SHOW_EVERY+1):
            s,rc,re,cap,loss = play_episode_and_learn(env, "chaser",
                                    chaser, evader, pf_ch, pf_ev, replay, schedule, hycfg)
            steps_acc += s; rc_acc += rc; re_acc += re; cap_acc += int(cap); avg_loss += loss; nloss += 1
            if (ep % schedule.PRINT_EVERY)==0:
                print(f"[Chaser] outer {outer} {ep}/{schedule.SHOW_EVERY} eps={chaser.eps:.3f} loss={(avg_loss/max(1,nloss)):.4f} replay={len(replay)}")

            if (ep % schedule.EVAL_EVERY)==0 and len(ep_pool)<schedule.POOL_LIMIT:
                info = rollout_greedy(env, "chaser", chaser, evader, pf_ch, schedule.MAX_STEPS)
                ep_pool.append(info)

        save_applied("chaser", chaser)
        stamp=time.strftime("%Y%m%d-%H%M%S")
        stats = {
            "phase":"chaser","outer":outer,"episodes":schedule.SHOW_EVERY,"timestamp":stamp,
            "avg_steps": steps_acc/schedule.SHOW_EVERY,
            "avg_reward_chaser": rc_acc/schedule.SHOW_EVERY,
            "avg_reward_evader": re_acc/schedule.SHOW_EVERY,
            "capture_rate": cap_acc/schedule.SHOW_EVERY,
            "epsilon_chaser": chaser.eps, "epsilon_evader": evader.eps,
            "avg_loss": (avg_loss/max(1,nloss)), "elapsed_sec": time.time()-t0,
            "replay_size": len(replay)
        }
        save_phase_log("chaser", outer, stamp, stats)
        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10*len(ep_pool_sorted)))
            top = ep_pool_sorted[:k]
            save_pool_json("chaser", outer, stamp, top)
            save_significant_json("chaser", outer, stamp, top)

        # ---------------- Phase B: Evader update ---------------- #
        ep_pool=[]; steps_acc=0; rc_acc=0.0; re_acc=0.0; cap_acc=0; avg_loss=0.0; nloss=0
        t0=time.time()
        for ep in range(1, schedule.SHOW_EVERY+1):
            s,rc,re,cap,loss = play_episode_and_learn(env, "evader",
                                    evader, chaser, pf_ev, pf_ch, replay, schedule, hycfg)
            steps_acc += s; rc_acc += rc; re_acc += re; cap_acc += int(cap); avg_loss += loss; nloss += 1
            if (ep % schedule.PRINT_EVERY)==0:
                print(f"[Evader] outer {outer} {ep}/{schedule.SHOW_EVERY} eps={evader.eps:.3f} loss={(avg_loss/max(1,nloss)):.4f} replay={len(replay)}")
            if (ep % schedule.EVAL_EVERY)==0 and len(ep_pool)<schedule.POOL_LIMIT:
                info = rollout_greedy(env, "evader", evader, chaser, pf_ev, schedule.MAX_STEPS)
                ep_pool.append(info)

        save_applied("evader", evader)
        stamp=time.strftime("%Y%m%d-%H%M%S")
        stats = {
            "phase":"evader","outer":outer,"episodes":schedule.SHOW_EVERY,"timestamp":stamp,
            "avg_steps": steps_acc/schedule.SHOW_EVERY,
            "avg_reward_chaser": rc_acc/schedule.SHOW_EVERY,
            "avg_reward_evader": re_acc/schedule.SHOW_EVERY,
            "capture_rate": cap_acc/schedule.SHOW_EVERY,
            "epsilon_chaser": chaser.eps, "epsilon_evader": evader.eps,
            "avg_loss": (avg_loss/max(1,nloss)), "elapsed_sec": time.time()-t0,
            "replay_size": len(replay)
        }
        save_phase_log("evader", outer, stamp, stats)
        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10*len(ep_pool_sorted)))
            top = ep_pool_sorted[:k]
            save_pool_json("evader", outer, stamp, top)
            save_significant_json("evader", outer, stamp, top)

        # -------- Optional big-map evaluation -------- #
        if outer >= schedule.OUTER_SWITCH_BIGMAP:
            big = EnvConfig(grid_size=40, fov_radius=18)
            env_big = TwoAgentTagEnv(big)
            pf_tmp = TorchParticleFilter(hycfg, big.grid_size, big.fov_radius, seed=7, device=device)
            res_ch = rollout_greedy(env_big, "chaser", chaser, evader, pf_tmp, max_steps=1200)
            res_ev = rollout_greedy(env_big, "evader", evader, chaser, pf_tmp, max_steps=1200)
            with open(os.path.join(EVAL_DIR, f"bigmap_eval_outer{outer}_{stamp}.json"), "w", encoding="utf-8") as f:
                json.dump({"chaser":res_ch, "evader":res_ev, "grid":big.grid_size}, f, ensure_ascii=False, indent=2)

        outer += 1

# --------------------------- Main ------------------------------------------ #
if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train_forever(
        env_cfg=EnvConfig(),
        hycfg=HybridConfig(),
        schedule=TrainSchedule(
            SHOW_EVERY=1000,
            PRINT_EVERY=50,
            MAX_STEPS=1000,
            EVAL_EVERY=100,
            POOL_LIMIT=300,
            OUTER_SWITCH_BIGMAP=5
        )
    )
