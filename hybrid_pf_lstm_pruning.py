# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (coins + decoy + egocentric actions)
# Hybrid: Particle-Filter summary + DRQN (Dueling + GRU)
# - No images. JSON only (for Unreal Engine parsing)
# - Alternating self-play training (role by role)
# - Action commit (durations) honored: some actions persist multiple ticks
# - Curriculum test: after threshold, run 2v2 (40x40) evaluation with shared policies
# - Save:
#   * Every 1,000 episodes: phase log JSON (outputs/logs/)
#   * End of each phase: applied overwrite JSON + .pt (outputs/applied/), snapshot .pt (outputs/snapshots/)
#   * Top-10% (role-phase pool) meaningful episodes JSON (outputs/logs/)
#
# Safe defaults for speed:
#  - PF particles = 300 (adjust via HybridConfig)
#  - Greedy evaluation only when needed (reduced frequency)
#
# Tested with: Python 3.10+, PyTorch 2.x

import os, time, json, math, random, threading, shutil
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================ OUTPUT PATHS ================================= #
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
APPLIED_DIR = os.path.join(OUTPUT_DIR, "applied")
SNAP_DIR    = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(APPLIED_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================== UTIL ====================================== #
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

# =============================== CONFIG ==================================== #
@dataclass
class Config:
    # World / observation
    grid_size: int = 20
    fov_radius: int = 10
    obs_noise: float = 0.12
    slip: float = 0.04
    capture_radius: int = 0
    seed: int = 42

    # Reward shaping
    time_penalty: float = 0.0005
    k_distance: float  = 0.005
    k_los: float       = 0.002
    k_info: float      = 0.0005

    # Action costs
    action_cost_rot: float      = 0.0003
    action_cost_look: float     = 0.00035
    action_cost_focus: float    = 0.00045
    action_cost_strafe: float   = 0.00015
    action_cost_back: float     = 0.00015
    action_cost_spin: float     = 0.0003
    action_cost_take_corner: float = 0.0005
    action_cost_collect: float  = 0.0003
    action_cost_decoy: float    = 0.0008

    # Coin mission
    coin_count: int = 5
    coin_collect_ticks: int = 5
    r_collect_tick: float = 0.2
    r_collect_done: float = 1.0
    r_all_coins_bonus: float = 5.0

    # Coin-aware shaping
    r_coin_detect: float = 0.05
    k_coin_approach: float = 0.002
    r_coin_visible: float = 0.01
    r_duplicate_penalty: float = -0.05

    # Threat-aware shaping
    threat_radius: int = 3
    k_threat_avoid: float = 0.003

    # DECOY effect (Evader)
    decoy_duration: int = 3
    r_decoy_success: float = 0.5
    decoy_slip_add: float = 0.08
    decoy_noise_mult: float = 2.0

    # Episode guard & early-stop
    max_steps: int = 1000
    no_progress_patience: int = 120   # steps without events -> early stop (train only)

    # Action commit (durations, in ticks)
    # (You can tune these for your presentation semantics)
    action_commit: Dict[str,int] = None

    def __post_init__(self):
        if self.action_commit is None:
            self.action_commit = {
                "WAIT":1,
                "ROT+1":1, "ROT-1":1, "ROT+2":1, "ROT-2":1,
                "SPIN":1,
                "FORWARD":1,
                "STRAFE_L":1, "STRAFE_R":1,
                "BACKSTEP":1,
                "LOOK":1,             # LOOK effects already linger via look_bonus
                "FOCUS":2,            # Presentation-wise: hold 2 ticks
                "TAKE_CORNER":1,
                "COLLECT":1,
                "DECOY":1
            }

# =============================== ACTIONS =================================== #
ACTIONS: List[Tuple[str, Tuple]] = [
    ("WAIT", ()),
    ("ROT+1", (+1,)), ("ROT-1", (-1,)), ("ROT+2", (+2,)), ("ROT-2", (-2,)),
    ("SPIN", ()),
    ("FORWARD", (1,)),
    ("STRAFE_L", ("L", 1)), ("STRAFE_R", ("R", 1)),
    ("BACKSTEP", (1,)),
    ("LOOK", ()),
    ("FOCUS", ()),          # FOV/noise bonus handled in env + commit=2
    ("TAKE_CORNER", ()),
    ("COLLECT", ()),        # Evader only
    ("DECOY", ()),          # Evader only
]
NUM_ACTIONS = len(ACTIONS)
assert NUM_ACTIONS == 15

# =============================== ENV (1v1) ================================= #
class TwoAgentTagEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser": 0, "evader": 0}
        self.coins = set(); self.coin_ticks = {}
        self.evader_collected = 0
        self.coins_seen = set()
        self.decoy_timer = 0

    def reset(self):
        while True:
            c = (random.randrange(self.n), random.randrange(self.n))
            e = (random.randrange(self.n), random.randrange(self.n))
            if manhattan(c, e) >= self.n//2: break
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

    def _apply_action(self, pos, face_idx, who: str, a_idx: int, other_pos):
        kind, param = ACTIONS[a_idx]
        n = self.n; new_pos, new_face = pos, face_idx

        if kind == "WAIT":
            pass
        elif kind.startswith("ROT"):
            step = param[0]; new_face = (new_face + step) % 8
        elif kind == "SPIN":
            new_face = (new_face + 4) % 8
        elif kind == "FORWARD":
            step = param[0]; fdx, fdy = dir_to_vec(new_face)
            for _ in range(step): new_pos = add_pos(new_pos, (fdx,fdy), n)
        elif kind == "STRAFE_L":
            sdx, sdy = dir_to_vec(new_face - 2)
            new_pos = add_pos(new_pos, (sdx,sdy), n)
        elif kind == "STRAFE_R":
            sdx, sdy = dir_to_vec(new_face + 2)
            new_pos = add_pos(new_pos, (sdx,sdy), n)
        elif kind == "BACKSTEP":
            bdx, bdy = dir_to_vec(new_face + 4)
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
            fdx, fdy = dir_to_vec(new_face)
            new_pos = add_pos(new_pos, (fdx,fdy), n)
        elif kind in ("COLLECT", "DECOY"):
            pass
        return new_pos, new_face

    def _nearest_coin_dist(self, epos) -> Tuple[Optional[Tuple[int,int]], int]:
        if not self.coins: return None, 99
        dists = [(p, manhattan(epos, p)) for p in self.coins]
        p, d = min(dists, key=lambda x: x[1])
        return p, d

    def step(self, a_c: int, a_e: int, eval_mode=False):
        (c, fc, e, fe) = self.state
        prev_d = manhattan(c, e)

        # timers
        if self.look_bonus["chaser"] > 0: self.look_bonus["chaser"] -= 1
        if self.look_bonus["evader"] > 0: self.look_bonus["evader"] -= 1
        if self.decoy_timer > 0: self.decoy_timer -= 1

        slip_c = self.cfg.slip + (self.cfg.decoy_slip_add if self.decoy_timer > 0 else 0.0)
        slip_e = self.cfg.slip

        if not eval_mode:
            if random.random() < slip_c: a_c = random.randrange(NUM_ACTIONS)
            if random.random() < slip_e: a_e = random.randrange(NUM_ACTIONS)

        c_next, fc_next = self._apply_action(c, fc, "chaser", a_c, e)
        e_next, fe_next = self._apply_action(e, fe, "evader", a_e, c)

        # Decoy
        r_decoy = 0.0
        if ACTIONS[a_e][0] == "DECOY":
            d_now = manhattan(e_next, c_next)
            if d_now <= self.cfg.fov_radius:
                self.decoy_timer = self.cfg.decoy_duration
                r_decoy = self.cfg.r_decoy_success

        # Coin
        r_collect_tick = 0.0; r_collect_done = 0.0; dup_penalty = 0.0
        if ACTIONS[a_e][0] == "COLLECT":
            if e_next in self.coins:
                self.coin_ticks[e_next] += 1
                r_collect_tick = self.cfg.r_collect_tick
                if self.coin_ticks[e_next] >= self.cfg.coin_collect_ticks:
                    self.coins.remove(e_next); del self.coin_ticks[e_next]
                    self.evader_collected += 1
                    r_collect_done = self.cfg.r_collect_done
            else:
                dup_penalty = self.cfg.r_duplicate_penalty

        self.state = (c_next, fc_next, e_next, fe_next)

        done_capture = (manhattan(c_next, e_next) <= self.cfg.capture_radius)
        done_allcoins = (self.evader_collected >= self.cfg.coin_count)
        done = done_capture or done_allcoins

        # Base rewards
        r_c = 1.0 if done_capture else 0.0
        r_e = (-1.0 if done_capture else 0.01)

        # Coin shaping
        nearest_p, nearest_d_now = self._nearest_coin_dist(e_next)
        if nearest_p is not None and manhattan(e_next, nearest_p) <= self.cfg.fov_radius:
            if nearest_p not in self.coins_seen:
                r_e += self.cfg.r_coin_detect
                self.coins_seen.add(nearest_p)
            r_e += self.cfg.r_coin_visible
        _, nearest_d_prev = self._nearest_coin_dist(e)
        if nearest_d_prev != 99 and nearest_d_now != 99:
            r_e += self.cfg.k_coin_approach * (nearest_d_prev - nearest_d_now)
        r_e += r_collect_tick + r_collect_done + r_decoy + dup_penalty
        if done_allcoins: r_e += self.cfg.r_all_coins_bonus

        # Generic shaping (distance/LOS/time)
        r_c -= self.cfg.time_penalty; r_e -= self.cfg.time_penalty
        new_d  = manhattan(c_next, e_next)
        delta_d = prev_d - new_d
        r_c += self.cfg.k_distance * (delta_d)
        r_e += self.cfg.k_distance * (-delta_d)
        new_far = new_d > self.cfg.fov_radius
        if not new_far: r_c += self.cfg.k_los
        else:           r_e += self.cfg.k_los

        kind_c, _ = ACTIONS[a_c]; kind_e, _ = ACTIONS[a_e]
        if kind_c == "LOOK":   r_c += self.cfg.k_info
        if kind_c == "FOCUS":  r_c += 0.75 * self.cfg.k_info
        if kind_e == "LOOK":   r_e += self.cfg.k_info
        if kind_e == "FOCUS":  r_e += 0.75 * self.cfg.k_info

        threat = (new_d <= self.cfg.threat_radius)
        if threat and kind_e not in ("COLLECT", "DECOY"):
            r_e += self.cfg.k_threat_avoid
        elif threat and kind_e == "COLLECT":
            r_e -= self.cfg.k_threat_avoid

        def act_cost(kind: str) -> float:
            if   kind.startswith("ROT"):  return self.cfg.action_cost_rot
            elif kind == "LOOK":          return self.cfg.action_cost_look
            elif kind == "FOCUS":         return self.cfg.action_cost_focus
            elif kind.startswith("STRAFE"):return self.cfg.action_cost_strafe
            elif kind == "BACKSTEP":      return self.cfg.action_cost_back
            elif kind == "SPIN":          return self.cfg.action_cost_spin
            elif kind == "TAKE_CORNER":   return self.cfg.action_cost_take_corner
            elif kind == "COLLECT":       return self.cfg.action_cost_collect
            elif kind == "DECOY":         return self.cfg.action_cost_decoy
            elif kind == "FORWARD":       return 0.0
            else:                         return 0.0
        r_c -= act_cost(kind_c); r_e -= act_cost(kind_e)

        if done_capture:
            r_c = 1.0; r_e = -1.0

        # events (for early stop detection)
        events = {
            "capture": done_capture,
            "coin_collected": (r_collect_done > 0.0),
            "decoy": (ACTIONS[a_e][0] == "DECOY" and r_decoy > 0.0),
            "distance_change": (delta_d != 0),
        }

        return self.state, (r_c, r_e), done, events

    # Observation (POMDP)
    def _dist_bin(self, d, cuts=(1,2,4,6,9,13,18,24)):
        for i,c in enumerate(cuts):
            if d <= c: return i
        return len(cuts)

    def _coin_dist_bin(self, d):
        if d <= 2: return 0
        if d <= 5: return 1
        if d <= 9: return 2
        return 3

    def observe(self, for_chaser: bool, eval_mode=False) -> Tuple[int,int,bool,int,int,int]:
        (c, fc, e, fe) = self.state
        me_pos, me_face = (c, fc) if for_chaser else (e, fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"

        bonus_turns = self.look_bonus[key]
        extra_noise_mult = (self.cfg.decoy_noise_mult if (for_chaser and self.decoy_timer>0) else 1.0)

        if not eval_mode and bonus_turns >= 2:
            eff_fov = int(self.cfg.fov_radius * 1.25)
            base_noise = self.cfg.obs_noise * 0.5
        elif not eval_mode and bonus_turns >= 1:
            eff_fov = int(self.cfg.fov_radius * 1.5)
            base_noise = self.cfg.obs_noise * 0.5
        else:
            eff_fov = self.cfg.fov_radius
            base_noise = self.cfg.obs_noise
        eff_noise = 0.0 if eval_mode else base_noise * extra_noise_mult

        dx, dy = ot_pos[0]-me_pos[0], ot_pos[1]-me_pos[1]
        d = abs(dx)+abs(dy); far = d > eff_fov
        if far:
            ang_bin = 0; dist_bin = 9
        else:
            ang_bin = angle_bin_from_dxdy16(dx, dy)
            dist_bin = self._dist_bin(d)
            if random.random() < eff_noise: ang_bin = (ang_bin + random.choice([-1,1,2])) % 16
            if random.random() < eff_noise: dist_bin = max(0, min(8, dist_bin + random.choice([-1,1])))
        my_face_bin = me_face

        # coin features (w.r.t. evader)
        nearest_p, nearest_d = self._nearest_coin_dist(e)
        coin_dist_bin = 3
        see_coin_flag = 0
        if nearest_p is not None:
            coin_dist_bin = self._coin_dist_bin(nearest_d)
            if nearest_d <= eff_fov:
                see_coin_flag = 1
                if not eval_mode and random.random() < eff_noise:
                    see_coin_flag = 1 - see_coin_flag
        return (ang_bin, dist_bin, far, my_face_bin, coin_dist_bin, see_coin_flag)

# ============================== BELIEF (bins) =============================== #
class BeliefIndexer:
    def __init__(self, na=16, nd=10, nf=8, nc=4, ns=2):
        self.na, self.nd, self.nf, self.nc, self.ns = na, nd, nf, nc, ns
    def index(self, obs: Tuple[int,int,bool,int,int,int]) -> int:
        ang, dist, far, face, cbin, see = obs
        dcode = 9 if far else dist
        return (((((face*self.nd) + dcode)*self.na + (ang % self.na))*self.nc + cbin)*self.ns + see)
    @property
    def n_bins(self): return self.na*self.nd*self.nf*self.nc*self.ns  # 10240

# ============================ HYBRID (PF + DRQN) ============================ #
@dataclass
class HybridConfig:
    pf_num_particles: int = 300         # ↓ speed-up
    pf_resample_every: int = 3
    pf_process_noise: float = 0.75
    pf_likelihood_ang_sigma: float = 1.0
    pf_likelihood_dist_sigma: float = 1.0
    pf_topk_modes: int = 3

    obs_dim_onehot: int = 41
    belief_feat_dim: int = 11          # 2(mean)+1(dist)+1(vartrace)+1(entropy)+2*topk(=6)

    hidden: int = 128
    n_actions: int = 15
    gamma: float = 0.997
    lr: float = 3e-4
    n_step: int = 1

    burn_in: int = 20
    unroll: int = 40
    target_update: int = 2000

    replay_capacity: int = 50_000
    batch_size: int = 16

    eps_start: float = 0.2
    eps_final: float = 0.02
    eps_decay: float = 0.9995

def obs_to_onehot(obs: Tuple[int,int,bool,int,int,int]) -> np.ndarray:
    ang, dist, far, face, coin_bin, see = obs
    v = []
    def oh(i, n):
        a = np.zeros(n, dtype=np.float32); a[int(i)%n] = 1.0; return a
    v.append(oh(ang, 16))                           # 16
    v.append(oh(9 if far else dist, 10))            # 10
    v.append(np.array([1.0 if far else 0.0], np.float32)) # 1
    v.append(oh(face, 8))                           # 8
    v.append(oh(coin_bin, 4))                       # 4
    v.append(oh(see, 2))                            # 2
    return np.concatenate(v, axis=0)                # 41

class ParticleFilter:
    def __init__(self, cfg: HybridConfig, grid_size:int, fov_radius:int, seed:int=0):
        self.cfg = cfg; self.n = grid_size; self.fov = fov_radius
        self.rng = np.random.default_rng(seed)
        self.num = cfg.pf_num_particles
        self.p = None; self.w = None; self._steps = 0
        self.reset()

    def reset(self):
        low = -self.n + 1; high = self.n - 1
        self.p = self.rng.integers(low, high+1, size=(self.num, 2)).astype(np.float32)
        self.w = np.ones(self.num, dtype=np.float32)/self.num
        self._steps = 0

    def predict(self, my_move: Tuple[int,int]=(0,0)):
        dx, dy = my_move
        self.p[:,0] -= dx; self.p[:,1] -= dy
        jitter = self.rng.normal(0.0, self.cfg.pf_process_noise, size=self.p.shape).astype(np.float32)
        self.p += jitter
        self.p[:,0] = np.clip(self.p[:,0], -self.n+1, self.n-1)
        self.p[:,1] = np.clip(self.p[:,1], -self.n+1, self.n-1)
        self._steps += 1

    def _angle_bin16(self, dx, dy):
        if dx==0 and dy==0: return 0
        a = math.atan2(dy, dx); 
        if a<0: a+=2*math.pi
        return int((a + (2*math.pi/16)/2)//(2*math.pi/16)) % 16

    def _dist_bin10(self, d):
        cuts = [1,2,4,6,9,13,18,24]
        for i,c in enumerate(cuts):
            if d<=c: return i
        return len(cuts)

    def weight_update(self, obs: Tuple[int,int,bool,int,int,int]):
        ang, dist, far, _, _, _ = obs
        dx = self.p[:,0]; dy = self.p[:,1]
        dist_mh = np.abs(dx) + np.abs(dy)
        pred_far = dist_mh > self.fov
        pred_ang = np.array([self._angle_bin16(dx[i], dy[i]) if not pred_far[i] else 0
                             for i in range(self.num)], dtype=np.int32)
        pred_dist = np.array([9 if pred_far[i] else self._dist_bin10(dist_mh[i])
                              for i in range(self.num)], dtype=np.int32)

        def circ(a,b):
            d = abs((a-b)%16); return min(d, 16-d)
        ang_err = np.array([circ(pred_ang[i], ang) for i in range(self.num)], dtype=np.float32)
        dist_err = np.abs(pred_dist - (9 if far else dist)).astype(np.float32)

        s1, s2 = self.cfg.pf_likelihood_ang_sigma, self.cfg.pf_likelihood_dist_sigma
        like = np.exp(-(ang_err**2)/(2*s1**2)) * np.exp(-(dist_err**2)/(2*s2**2)) + 1e-8
        self.w *= like
        s = float(self.w.sum(dtype=np.float64))
        if (not np.isfinite(s)) or s<=0.0:
            self.w.fill(1.0/self.num)
        else:
            self.w = (self.w / s).astype(np.float32)
        if (self._steps % self.cfg.pf_resample_every)==0:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.num
        positions = (np.arange(N, dtype=np.float64) + float(self.rng.random())) / float(N)
        cumsum = np.cumsum(self.w, dtype=np.float64); cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions, side='left')
        idx = np.minimum(idx, N-1)
        self.p = self.p[idx]; self.w.fill(1.0/N)

    def summarize(self, topk:int=None) -> np.ndarray:
        if topk is None: topk = self.cfg.pf_topk_modes
        mean = (self.w[:,None]*self.p).sum(axis=0)
        var_trace = (self.w[:,None]*((self.p-mean)**2)).sum(axis=0).sum()
        mean_dist = (np.abs(self.p).sum(axis=1)*self.w).sum()
        w = np.clip(self.w, 1e-12, 1.0)
        entropy = float(-(w*np.log(w)).sum() / math.log(len(w)))
        idx = np.argsort(-self.w)[:topk]; top = self.p[idx]
        s = float(self.n)
        feats = [mean[0]/s, mean[1]/s, mean_dist/(2*s), min(1.0, var_trace/(s*s)), entropy]
        for i in range(topk):
            if i < len(top): feats.extend([float(top[i,0]/s), float(top[i,1]/s)])
            else: feats.extend([0.0, 0.0])
        return np.array(feats, dtype=np.float32)

class DRQN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x_seq: torch.Tensor, h0: Optional[torch.Tensor]=None):
        z = self.enc(x_seq)
        out, hT = self.gru(z, h0)
        V = self.val(out)
        A = self.adv(out)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hT

class SeqReplay:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.data: List[Dict[str,np.ndarray]] = []
        self.ptr = 0
    def push_episode(self, traj: Dict[str, np.ndarray]):
        if len(self.data) < self.capacity:
            self.data.append(traj)
        else:
            self.data[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.capacity
    def sample_batch(self, batch:int, burn_in:int, unroll:int):
        if len(self.data) < batch: return None
        out = []
        seqs = random.sample(self.data, batch)
        need = burn_in + unroll + 1
        for ep in seqs:
            T = len(ep["act"])
            if T < need: return None
            start = random.randint(0, T-need)
            sl = slice(start, start+need)
            out.append({k: v[sl] for k,v in ep.items()})
        return out

class HybridAgent:
    def __init__(self, hycfg: HybridConfig, device: str="cpu"):
        self.cfg = hycfg
        in_dim = hycfg.obs_dim_onehot + hycfg.belief_feat_dim
        self.net = DRQN(in_dim, hycfg.hidden, hycfg.n_actions).to(device)
        self.tgt = DRQN(in_dim, hycfg.hidden, hycfg.n_actions).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=hycfg.lr)
        self.device = device
        self.eps = hycfg.eps_start
        self.step_count = 0

    def decay_eps(self):
        self.eps = max(self.cfg.eps_final, self.eps*self.cfg.eps_decay)

    def q_eval(self, x: np.ndarray, h: Optional[torch.Tensor]):
        xt = torch.from_numpy(x).float().to(self.device).view(1,1,-1)
        with torch.no_grad():
            q, h2 = self.net(xt, h)
        return q, h2

    def act_eps(self, x: np.ndarray, h: Optional[torch.Tensor]):
        q, h2 = self.q_eval(x, h)
        if random.random() < self.eps:
            a = random.randrange(q.shape[-1])
        else:
            a = int(torch.argmax(q, dim=-1).item())
        return a, h2

    def act_greedy(self, x: np.ndarray, h: Optional[torch.Tensor]):
        q, h2 = self.q_eval(x, h)
        a = int(torch.argmax(q, dim=-1).item())
        return a, h2

    def train_batch(self, batch: List[Dict[str,np.ndarray]], gamma: float, burn:int, unroll:int):
        if batch is None: return 0.0
        device = self.device
        B = len(batch)

        def to_t(name):
            X = np.stack([b[name] for b in batch],0)
            return torch.from_numpy(X).float().to(device)

        x = to_t("x")                      # (B,T,dim)
        a = torch.from_numpy(np.stack([b["act"] for b in batch],0)).long().to(device)
        r = torch.from_numpy(np.stack([b["rew"] for b in batch],0)).float().to(device)
        d = torch.from_numpy(np.stack([b["done"] for b in batch],0)).float().to(device)

        with torch.no_grad():
            _, h = self.net(x[:, :burn, :])
        q_on, _ = self.net(x[:, burn:-1, :], h)                      # (B, unroll, A)
        q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_on_next, _  = self.net(x[:, burn+1:, :], h)
            a_star = torch.argmax(q_on_next, dim=-1)
            q_tgt_next, _ = self.tgt(x[:, burn+1:, :], h)
            q_next = q_tgt_next.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)

        target = r[:, burn:burn+unroll] + gamma * (1.0 - d[:, burn:burn+unroll]) * q_next
        loss = F.smooth_l1_loss(q_sel, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.step_count += 1
        if (self.step_count % self.cfg.target_update) == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        self.decay_eps()
        return float(loss.item())

# ============================ ACTION COMMIT UTIL ============================ #
def my_move_delta_from_action(action_idx:int, my_face_bin:int) -> Tuple[int,int]:
    def vec(f): return DIRS8[f%8]
    if action_idx==6:   vx,vy = vec(my_face_bin)         # FORWARD
    elif action_idx==7: vx,vy = vec((my_face_bin-2)%8)   # STRAFE_L
    elif action_idx==8: vx,vy = vec((my_face_bin+2)%8)   # STRAFE_R
    elif action_idx==9: vx,vy = vec((my_face_bin+4)%8)   # BACKSTEP
    else:               vx,vy = (0,0)
    return (vx,vy)

def committed_action(next_new_action_idx: int,
                     last_action_idx: Optional[int],
                     commit_left: int,
                     action_commit_map: Dict[str,int]) -> Tuple[int,int,Optional[int]]:
    """
    Return (action_idx_to_execute, new_commit_left, maybe_reset_last_action_idx).
    """
    if commit_left > 0 and last_action_idx is not None:
        # keep last action
        return last_action_idx, commit_left-1, last_action_idx
    # start a new commit window
    kind = ACTIONS[next_new_action_idx][0]
    dur = action_commit_map.get(kind, 1)
    return next_new_action_idx, max(0, dur-1), next_new_action_idx

# ============================ EPISODE (train) ============================== #
def play_episode_and_learn(env: TwoAgentTagEnv,
                           who: str,
                           agent_self: HybridAgent,
                           agent_other: HybridAgent,
                           pf_self: ParticleFilter,
                           pf_other: Optional[ParticleFilter],
                           replay: SeqReplay,
                           cfg: Config,
                           hycfg: HybridConfig,
                           train_mode=True):
    assert who in ("chaser","evader")
    env.reset()
    pf_self.reset()
    if pf_other is not None: pf_other.reset()

    xs, acts, rews, dones = [], [], [], []
    h_self = None; h_oth = None

    # action commit trackers
    last_a_self: Optional[int] = None; commit_left_self = 0
    last_a_oth:  Optional[int] = None; commit_left_oth  = 0

    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=not train_mode)
    ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=not train_mode)

    sum_r_c = 0.0; sum_r_e = 0.0
    steps = 0; captured=False

    no_prog = 0
    def any_event(ev): return ev["capture"] or ev["coin_collected"] or ev["decoy"] or ev["distance_change"]

    while steps < cfg.max_steps:
        x_self = np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)
        if pf_other is not None:
            x_oth  = np.concatenate([obs_to_onehot(ob_o), pf_other.summarize()], axis=0)
        else:
            x_oth  = np.concatenate([obs_to_onehot(ob_o), np.zeros(hycfg.belief_feat_dim, np.float32)], axis=0)

        # action proposal
        a_prop_self,   h_self = (agent_self.act_eps(x_self, h_self) if train_mode else agent_self.act_greedy(x_self, h_self))
        a_prop_other,  h_oth  = agent_other.act_greedy(x_oth, h_oth)

        # apply commit windows
        a_s, commit_left_self, last_a_self = committed_action(a_prop_self, last_a_self, commit_left_self, cfg.action_commit)
        a_o, commit_left_oth,  last_a_oth  = committed_action(a_prop_other, last_a_oth,  commit_left_oth,  cfg.action_commit)

        # env step
        if who=="chaser":
            _, (r_c, r_e), done, evs = env.step(a_s, a_o, eval_mode=not train_mode)
            r_who = r_c
        else:
            _, (r_c, r_e), done, evs = env.step(a_o, a_s, eval_mode=not train_mode)
            r_who = r_e

        sum_r_c += r_c; sum_r_e += r_e

        # PF updates (self)
        my_face_bin = ob_s[3]
        pf_self.predict(my_move=my_move_delta_from_action(a_s if who=="chaser" else a_o, my_face_bin))
        ob_s_next = env.observe(for_chaser=(who=="chaser"), eval_mode=not train_mode)
        pf_self.weight_update(ob_s_next)

        # store transition (who-perspective)
        xs.append(np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0))
        acts.append(a_s if who=="chaser" else a_o)
        rews.append(r_who); dones.append(1.0 if done else 0.0)

        # PF other (optional light model; for speed set pf_other=None)
        if pf_other is not None:
            my_face_bin_o = ob_o[3]
            pf_other.predict(my_move=my_move_delta_from_action(a_o if who=="chaser" else a_s, my_face_bin_o))
            ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=not train_mode)
            pf_other.weight_update(ob_o)
        else:
            ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=not train_mode)

        ob_s = ob_s_next
        steps += 1

        # early stop (train only)
        if train_mode:
            if any_event(evs): no_prog = 0
            else: no_prog += 1
            if no_prog >= cfg.no_progress_patience:
                break

        if done:
            captured=True
            break

    traj = {
        "x":   np.stack(xs,0).astype(np.float32),
        "act": np.array(acts, dtype=np.int64),
        "rew": np.array(rews, dtype=np.float32),
        "done":np.array(dones, dtype=np.float32)
    }
    replay.push_episode(traj)

    # one minibatch
    batch = replay.sample_batch(batch=hycfg.batch_size, burn_in=hycfg.burn_in, unroll=hycfg.unroll)
    loss = agent_self.train_batch(batch, hycfg.gamma, hycfg.burn_in, hycfg.unroll) if (train_mode and batch) else 0.0

    return {
        "steps": steps,
        "sum_r_c": sum_r_c,
        "sum_r_e": sum_r_e,
        "captured": captured,
        "loss": float(loss)
    }

# ============================ EVAL (greedy) ================================ #
def eval_episode(env: TwoAgentTagEnv,
                 who: str,
                 agent_self: HybridAgent,
                 agent_other: HybridAgent,
                 pf_self: ParticleFilter,
                 cfg: Config,
                 hycfg: HybridConfig):
    return play_episode_and_learn(env, who, agent_self, agent_other, pf_self, None,
                                  replay=SeqReplay(1), cfg=cfg, hycfg=hycfg, train_mode=False)

# ============================ LOGGING / JSON =============================== #
def save_applied_json(role:str, agent:HybridAgent, path_pt:str, path_json:str):
    meta = {
        "role": role,
        "arch": "DRQN(Dueling)-GRU",
        "obs_dim": agent.cfg.obs_dim_onehot,
        "belief_dim": agent.cfg.belief_feat_dim,
        "hidden": agent.cfg.hidden,
        "n_actions": agent.cfg.n_actions,
        "gamma": agent.cfg.gamma,
        "eps": round(agent.eps, 5),
        "step_count": int(agent.step_count),
        "checkpoint": os.path.basename(path_pt),
        "checkpoint_fullpath": os.path.abspath(path_pt),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def save_phase_log_json(role:str, outer:int, ep_start:int, ep_end:int, agg:dict):
    # ep_start..ep_end inclusive block summary (e.g., every 1,000)
    name = f"log_{role}_outer{outer}_ep{ep_start}-{ep_end}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path = os.path.join(LOGS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

def summarize_episode_for_unreal(stats: dict, who:str, extra:dict=None):
    # Minimal schema for UE parsing (extend as needed)
    # stats from play_episode_and_learn(): steps,sum_r_c,sum_r_e,captured,loss
    d = {
        "who": who,
        "steps": int(stats["steps"]),
        "reward_chaser": float(stats["sum_r_c"]),
        "reward_evader": float(stats["sum_r_e"]),
        "captured": bool(stats["captured"])
    }
    if extra: d.update(extra)
    return d

def filter_meaningful(episodes: List[dict], role:str):
    # role-specific filters
    if role == "chaser":
        # meaningful: captured True or large positive r_c
        return [e for e in episodes if e.get("captured", False) or e.get("reward_chaser",0.0) > 0.5]
    else:
        # evader: collected coins or long survival or high r_e
        return [e for e in episodes if (e.get("coins",0)>0) or (e.get("steps",0)>400) or e.get("reward_evader",0.0) > 1.0]

# ============================= CURRICULUM EVAL ============================= #
# 간단 2v2 평가: 같은 정책(가중치 공유)로 각각의 유닛이 독립적으로 행동 선택.
# 관찰은 가장 가까운 상대 기준으로 구성(빠른 프로토타입).
class TeamTagEvalEnv:
    def __init__(self, cfg: Config, n_team:int=2, grid:int=40):
        # lightweight eval env
        self.cfg = Config(**{**cfg.__dict__, "grid_size": grid})
        self.n = grid
        self.n_team = n_team
        self.reset()

    def reset(self):
        # chasers & evaders spawn far apart, coins for evader team shared
        self.ch = []
        self.ev = []
        for _ in range(self.n_team):
            self.ch.append((random.randrange(self.n), random.randrange(self.n), random.randrange(8)))
            self.ev.append((random.randrange(self.n), random.randrange(self.n), random.randrange(8)))
        self.ev_collected = 0
        return True

    def nearest(self, me_pos, others):
        # manhattan nearest
        dmin=10**9; best=None
        for (x,y,_) in others:
            d = manhattan(me_pos, (x,y))
            if d<dmin: dmin=d; best=(x,y)
        return best, dmin

    def observe_one(self, me:Tuple[int,int,int], others:List[Tuple[int,int,int]]):
        (x,y,f) = me
        tgt,_ = self.nearest((x,y), others)
        dx,dy = tgt[0]-x, tgt[1]-y
        d = abs(dx)+abs(dy); far = d > self.cfg.fov_radius
        ang = angle_bin_from_dxdy16(dx,dy) if not far else 0
        dist_bin = 9 if far else TwoAgentTagEnv._dist_bin(self, d)
        coin_bin, see_coin = 3, 0
        return (ang, dist_bin, far, f, coin_bin, see_coin)

    def step_one(self, me, a_idx, target_pos):
        (x,y,f) = me
        kind,_ = ACTIONS[a_idx]
        if kind.startswith("ROT"):
            step = ACTIONS[a_idx][1][0]; f=(f+step)%8
        elif kind=="SPIN": f=(f+4)%8
        elif kind=="FORWARD":
            vx,vy = dir_to_vec(f); x,y = add_pos((x,y),(vx,vy), self.n)
        elif kind=="STRAFE_L":
            vx,vy = dir_to_vec((f-2)%8); x,y = add_pos((x,y),(vx,vy), self.n)
        elif kind=="STRAFE_R":
            vx,vy = dir_to_vec((f+2)%8); x,y = add_pos((x,y),(vx,vy), self.n)
        elif kind=="BACKSTEP":
            vx,vy = dir_to_vec((f+4)%8); x,y = add_pos((x,y),(vx,vy), self.n)
        elif kind=="TAKE_CORNER":
            dx,dy = target_pos[0]-x, target_pos[1]-y
            desired = angle_bin_from_dxdy16(dx,dy)%8
            diff=(desired-f)%8; f=(f+(+1 if diff<=4 else -1))%8
            vx,vy = dir_to_vec(f); x,y = add_pos((x,y),(vx,vy), self.n)
        # LOOK/FOCUS/DECOY/COLLECT: no-op for this lightweight eval shell
        return (x,y,f)

    def step(self, a_ch: List[int], a_ev: List[int]):
        # apply actions
        new_ch = []; new_ev=[]
        for i,(x,y,f) in enumerate(self.ch):
            tgt,_ = self.nearest((x,y), [(ex,ey,ef) for (ex,ey,ef) in self.ev])
            new_ch.append(self.step_one((x,y,f), a_ch[i], tgt))
        for i,(x,y,f) in enumerate(self.ev):
            tgt,_ = self.nearest((x,y), [(cx,cy,cf) for (cx,cy,cf) in self.ch])
            new_ev.append(self.step_one((x,y,f), a_ev[i], tgt))
        self.ch = new_ch; self.ev = new_ev

        # capture check (any close)
        captured_pairs = 0
        for (cx,cy,_) in self.ch:
            for (ex,ey,_) in self.ev:
                if manhattan((cx,cy), (ex,ey)) <= self.cfg.capture_radius:
                    captured_pairs += 1
        return captured_pairs

def curriculum_eval_2v2(agent_ch:HybridAgent, agent_ev:HybridAgent, base_cfg:Config, hycfg:HybridConfig,
                        episodes:int=10, max_steps:int=400):
    env = TeamTagEvalEnv(base_cfg, n_team=2, grid=40)
    out=[]
    for _ in range(episodes):
        env.reset()
        # PFs per unit omitted (lightweight test). Use zeros.
        hch=[None,None]; hev=[None,None]
        steps=0; captures=0
        while steps<max_steps:
            a_ch=[]; a_ev=[]
            for i in range(2):
                ob = env.observe_one(env.ch[i], env.ev)
                x = np.concatenate([obs_to_onehot(ob), np.zeros(hycfg.belief_feat_dim, np.float32)], axis=0)
                ai, hch[i] = agent_ch.act_greedy(x, hch[i]); a_ch.append(ai)
            for i in range(2):
                ob = env.observe_one(env.ev[i], env.ch)
                x = np.concatenate([obs_to_onehot(ob), np.zeros(hycfg.belief_feat_dim, np.float32)], axis=0)
                ai, hev[i] = agent_ev.act_greedy(x, hev[i]); a_ev.append(ai)
            captures += env.step(a_ch, a_ev)
            steps+=1
        out.append({"steps":steps, "captures":captures})
    return out

# =========================== TRAIN FOREVER LOOP =============================#
class StopFlag:
    def __init__(self): self.flag=False
def stop_listener(sf: StopFlag):
    try:
        s = input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s == 'q': sf.flag=True
    except: pass

def train_forever(
    cfg: Config = Config(),
    hycfg: HybridConfig = HybridConfig(),
    SHOW_EVERY=10000,        # episodes per phase for one role
    PRINT_EVERY=1000,        # console print block
    EVAL_POOL_EVERY=250,     # push eval episode into pool
    POOL_LIMIT=500,          # per phase
    LOG_BLOCK=1000,          # save JSON every 1000 eps
    CURRICULUM_AFTER_OUTER: Optional[int]=None,  # if set, after that outer index run 2v2 eval
    CURRICULUM_EVAL_EPISODES:int=10
):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = TwoAgentTagEnv(cfg)
    beliefs = BeliefIndexer()  # kept for compatibility (not used for learning)

    chaser = HybridAgent(hycfg, device=device)
    evader = HybridAgent(hycfg, device=device)

    # For speed, only self PF is used; other PF is None
    pf_ch = ParticleFilter(hycfg, cfg.grid_size, cfg.fov_radius, seed=0)
    pf_ev = ParticleFilter(hycfg, cfg.grid_size, cfg.fov_radius, seed=1)
    replay = SeqReplay(hycfg.replay_capacity)

    sf = StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    outer = 1
    while not sf.flag:
        # -------------------- Phase A: Chaser updates -------------------- #
        block_start = 1
        pool=[]
        agg_block = {"role":"chaser","outer":outer,"blocks":[]}
        ep_rewards=[]

        steps_acc = r_c_acc = r_e_acc = 0.0
        cap_cnt = 0; loss_acc=0.0; loss_cnt=0

        for ep in range(1, SHOW_EVERY+1):
            res = play_episode_and_learn(env,"chaser", chaser, evader, pf_ch, None, replay, cfg, hycfg, train_mode=True)
            steps_acc += res["steps"]; r_c_acc += res["sum_r_c"]; r_e_acc += res["sum_r_e"]; cap_cnt += int(res["captured"])
            loss_acc += res["loss"]; loss_cnt+=1
            ep_rewards.append(res["sum_r_c"])

            # pool push
            if (ep % EVAL_POOL_EVERY)==0 and len(pool)<POOL_LIMIT:
                ev = eval_episode(env,"chaser",chaser,evader,pf_ch,cfg,hycfg)
                pool.append(summarize_episode_for_unreal(ev,"chaser"))

            # block log save
            if (ep % LOG_BLOCK)==0:
                avg_loss = (loss_acc/max(1,loss_cnt))
                block = {
                    "episodes": [block_start, ep],
                    "avg_steps": steps_acc/LOG_BLOCK,
                    "avg_reward_chaser": r_c_acc/LOG_BLOCK,
                    "avg_reward_evader": r_e_acc/LOG_BLOCK,
                    "capture_rate": cap_cnt/LOG_BLOCK,
                    "avg_loss": avg_loss,
                    "epsilon": chaser.eps
                }
                save_phase_log_json("chaser", outer, block_start, ep, block)
                agg_block["blocks"].append(block)
                # reset block accumulators
                block_start = ep+1
                steps_acc = r_c_acc = r_e_acc = 0.0
                cap_cnt = 0; loss_acc=0.0; loss_cnt=0

            if sf.flag: break
            if (ep % PRINT_EVERY)==0:
                print(f"[Chaser] outer {outer} ep {ep}/{SHOW_EVERY} eps={chaser.eps:.3f}")

        if sf.flag: break

        # applied save
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pt_path = os.path.join(APPLIED_DIR, "chaser_policy.pt")
        torch.save(chaser.net.state_dict(), pt_path)
        save_applied_json("chaser", chaser, pt_path, os.path.join(APPLIED_DIR,"chaser_applied.json"))
        # snapshot
        torch.save(chaser.net.state_dict(), os.path.join(SNAP_DIR, f"chaser_outer{outer}_{stamp}.pt"))

        # top-10% meaningful episodes JSON
        if pool:
            pool_sorted = sorted(pool, key=lambda d: d["reward_chaser"], reverse=True)
            k = max(1, math.ceil(0.10*len(pool_sorted)))
            top_eps = pool_sorted[:k]
            # add role-specific extras (optionally attach action hist etc. if you log them)
            meaningful = filter_meaningful(top_eps, "chaser")
            with open(os.path.join(LOGS_DIR, f"top10p_meaningful_chaser_outer{outer}_{stamp}.json"), "w", encoding="utf-8") as f:
                json.dump({"outer":outer,"role":"chaser","episodes":meaningful}, f, ensure_ascii=False, indent=2)

        # -------------------- Phase B: Evader updates -------------------- #
        block_start = 1
        pool=[]
        steps_acc = r_c_acc = r_e_acc = 0.0
        cap_cnt = 0; loss_acc=0.0; loss_cnt=0

        for ep in range(1, SHOW_EVERY+1):
            res = play_episode_and_learn(env,"evader", evader, chaser, pf_ev, None, replay, cfg, hycfg, train_mode=True)
            steps_acc += res["steps"]; r_c_acc += res["sum_r_c"]; r_e_acc += res["sum_r_e"]; cap_cnt += int(res["captured"])
            loss_acc += res["loss"]; loss_cnt+=1

            if (ep % EVAL_POOL_EVERY)==0 and len(pool)<POOL_LIMIT:
                ev = eval_episode(env,"evader",evader,chaser,pf_ev,cfg,hycfg)
                # 여기서 코인 수집 추정치는 평과셸에선 없으므로 0으로 둠(원하면 학습 env 기반 별도 계산 추가)
                pool.append(summarize_episode_for_unreal(ev,"evader", extra={"coins":0}))

            if (ep % LOG_BLOCK)==0:
                avg_loss = (loss_acc/max(1,loss_cnt))
                block = {
                    "episodes": [block_start, ep],
                    "avg_steps": steps_acc/LOG_BLOCK,
                    "avg_reward_chaser": r_c_acc/LOG_BLOCK,
                    "avg_reward_evader": r_e_acc/LOG_BLOCK,
                    "capture_rate": cap_cnt/LOG_BLOCK,
                    "avg_loss": avg_loss,
                    "epsilon": evader.eps
                }
                save_phase_log_json("evader", outer, block_start, ep, block)
                block_start = ep+1
                steps_acc = r_c_acc = r_e_acc = 0.0
                cap_cnt = 0; loss_acc=0.0; loss_cnt=0

            if sf.flag: break
            if (ep % PRINT_EVERY)==0:
                print(f"[Evader] outer {outer} ep {ep}/{SHOW_EVERY} eps={evader.eps:.3f}")

        if sf.flag: break

        # applied save
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pt_path = os.path.join(APPLIED_DIR, "evader_policy.pt")
        torch.save(evader.net.state_dict(), pt_path)
        save_applied_json("evader", evader, pt_path, os.path.join(APPLIED_DIR,"evader_applied.json"))
        torch.save(evader.net.state_dict(), os.path.join(SNAP_DIR, f"evader_outer{outer}_{stamp}.pt"))

        if pool:
            pool_sorted = sorted(pool, key=lambda d: d["reward_evader"], reverse=True)
            k = max(1, math.ceil(0.10*len(pool_sorted)))
            top_eps = pool_sorted[:k]
            meaningful = filter_meaningful(top_eps, "evader")
            with open(os.path.join(LOGS_DIR, f"top10p_meaningful_evader_outer{outer}_{stamp}.json"), "w", encoding="utf-8") as f:
                json.dump({"outer":outer,"role":"evader","episodes":meaningful}, f, ensure_ascii=False, indent=2)

        # Optional curriculum evaluation (2v2 on 40x40) after certain outer
        if CURRICULUM_AFTER_OUTER is not None and outer >= CURRICULUM_AFTER_OUTER:
            cur = curriculum_eval_2v2(chaser, evader, cfg, hycfg, episodes=CURRICULUM_EVAL_EPISODES, max_steps=400)
            with open(os.path.join(LOGS_DIR, f"curriculum_eval_2v2_outer{outer}_{stamp}.json"), "w", encoding="utf-8") as f:
                json.dump({"outer":outer,"mode":"2v2_40x40","episodes":cur}, f, ensure_ascii=False, indent=2)

        outer += 1

# ================================ MAIN ===================================== #
if __name__ == "__main__":
    train_forever(
        cfg=Config(max_steps=1000, no_progress_patience=120),
        hycfg=HybridConfig(),
        SHOW_EVERY=10000,
        PRINT_EVERY=1000,
        EVAL_POOL_EVERY=250,
        POOL_LIMIT=500,
        LOG_BLOCK=1000,
        CURRICULUM_AFTER_OUTER=2,       # 예: 2번째 외부 페이즈 이후 2v2 평가 실행
        CURRICULUM_EVAL_EPISODES=10
    )
