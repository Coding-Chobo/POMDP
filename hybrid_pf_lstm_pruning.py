# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (coins + decoy + egocentric actions)
# Hybrid: Particle-Filter summary + GRU(DRQN)
# 요구사항 반영:
#  - POMDP 본질 유지: "진전 없음"은 조기 종료 대신 학습 시퀀스에서 **데이터 필터링**(프레임 스킵)으로 제외
#  - 에피소드 스텝가드: 1000
#  - 코인 중심 보상: evader per-step 생존보상 0, 수집/탐지 보상 상향
#  - LOOK 보너스: PF 엔트로피 감소(정보이득) 있을 때만 외부 shaping으로 부여
#  - 프루닝(pruning): 시각화용 정책그래프 사본에만 적용(학습/데이터에는 절대 적용 X)
#  - Best-episode 기반 정책그래프 + Top-10% 합성 PNG 저장
#
# 참고:
#  - Tabular Q 전면 제거; 함수근사 DRQN(Double+Dueling) 단일 파일
#  - PyTorch >= 2.x 권장
#  - SHOW_EVERY/PRINT_EVERY는 하단 main에서 쉽게 조정 가능

import os, time, json, math, random, threading, shutil
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= FFMPEG AUTO-DETECT (PNG only) ===================== #
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    common_candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\Desktop\TUK\졸작\연구주제\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in common_candidates:
        if os.path.exists(p):
            ffmpeg_path = p
            break
if ffmpeg_path:
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    print(f"[ffmpeg] ✅ detected at: {ffmpeg_path}")
else:
    print("[ffmpeg] ⚠ not found. (No animation; PNG only)")

# ============================ OUTPUT PATHS ================================= #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
APPLIED_DIR = os.path.join(OUTPUT_DIR, "applied")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
MEDIA_DIR = os.path.join(OUTPUT_DIR, "media")
os.makedirs(APPLIED_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

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

    # Reward shaping (환경 내부)
    time_penalty: float = 0.0005
    k_distance: float  = 0.005
    k_los: float       = 0.002
    k_info: float      = 0.0   # << LOOK/FOCUS 내부 보상은 0으로 두고, 외부 shaping에서 정보이득 시에만 부여

    # Action costs
    action_cost_wait: float     = 0.0
    action_cost_rot: float      = 0.0003
    action_cost_look: float     = 0.00035
    action_cost_focus: float    = 0.00045
    action_cost_strafe: float   = 0.00015
    action_cost_back: float     = 0.00015
    action_cost_spin: float     = 0.0003
    action_cost_take_corner: float = 0.0005
    action_cost_collect: float  = 0.0003
    action_cost_decoy: float    = 0.0008

    # Coin mission (코인 중심)
    coin_count: int = 5
    coin_collect_ticks: int = 5
    r_collect_tick: float = 0.5     # ↑
    r_collect_done: float = 2.0     # ↑
    r_all_coins_bonus: float = 10.0 # ↑

    # Coin-aware shaping
    r_coin_detect: float = 0.08
    k_coin_approach: float = 0.003
    r_coin_visible: float = 0.015
    r_duplicate_penalty: float = -0.05

    # Threat-aware shaping
    threat_radius: int = 3
    k_threat_avoid: float = 0.003

    # DECOY effect (Evader)
    decoy_duration: int = 3
    r_decoy_success: float = 0.5
    decoy_slip_add: float = 0.08
    decoy_noise_mult: float = 2.0

random.seed(42); np.random.seed(42); torch.manual_seed(42)

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

# =============================== ACTIONS =================================== #
ACTIONS: List[Tuple[str, Tuple]] = [
    ("WAIT", ()),
    ("ROT+1", (+1,)), ("ROT-1", (-1,)), ("ROT+2", (+2,)), ("ROT-2", (-2,)),
    ("SPIN", ()),
    ("FORWARD", (1,)),
    ("STRAFE_L", ("L", 1)), ("STRAFE_R", ("R", 1)),
    ("BACKSTEP", (1,)),
    ("LOOK", ()),
    ("FOCUS", ()),
    ("TAKE_CORNER", ()),
    ("COLLECT", ()),    # Evader only
    ("DECOY", ()),      # Evader only
]
NUM_ACTIONS = len(ACTIONS)

# =============================== ENV ======================================= #
class TwoAgentTagEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser": 0, "evader": 0}
        self.coins = set()
        self.coin_ticks = {}
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

        # DECOY
        r_decoy = 0.0
        if ACTIONS[a_e][0] == "DECOY":
            d_now = manhattan(e_next, c_next)
            if d_now <= self.cfg.fov_radius:
                self.decoy_timer = self.cfg.decoy_duration
                r_decoy = self.cfg.r_decoy_success

        # COIN
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
        r_e = (-1.0 if done_capture else 0.0)   # evader 생존 보상 0

        # Coin-aware shaping
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

        # Generic shaping
        r_c -= self.cfg.time_penalty; r_e -= self.cfg.time_penalty
        new_d  = manhattan(c_next, e_next)
        delta_d = prev_d - new_d
        r_c += self.cfg.k_distance * (delta_d)
        r_e += self.cfg.k_distance * (-delta_d)
        new_far = new_d > self.cfg.fov_radius
        if not new_far: r_c += self.cfg.k_los
        else:           r_e += self.cfg.k_los

        kind_c, _ = ACTIONS[a_c]; kind_e, _ = ACTIONS[a_e]
        # LOOK/FOCUS 보너스는 외부 shaping으로만 (k_info=0.0)

        def act_cost(kind: str) -> float:
            if   kind == "WAIT":        return 0.0
            elif kind.startswith("ROT"):return self.cfg.action_cost_rot
            elif kind == "LOOK":        return self.cfg.action_cost_look
            elif kind == "FOCUS":       return self.cfg.action_cost_focus
            elif kind.startswith("STRAFE"):  return self.cfg.action_cost_strafe
            elif kind == "BACKSTEP":    return self.cfg.action_cost_back
            elif kind == "SPIN":        return self.cfg.action_cost_spin
            elif kind == "TAKE_CORNER": return self.cfg.action_cost_take_corner
            elif kind == "COLLECT":     return self.cfg.action_cost_collect
            elif kind == "DECOY":       return self.cfg.action_cost_decoy
            elif kind == "FORWARD":     return 0.0
            else:                       return 0.0
        r_c -= act_cost(kind_c); r_e -= act_cost(kind_e)

        if done_capture:
            r_c = 1.0; r_e = -1.0

        return self.state, (r_c, r_e), done

    # -------------------- Observation (POMDP) -------------------- #
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
    pf_num_particles: int = 800
    pf_resample_every: int = 3
    pf_process_noise: float = 0.75
    pf_likelihood_ang_sigma: float = 1.0
    pf_likelihood_dist_sigma: float = 1.0
    pf_topk_modes: int = 3

    # obs one-hot dim: ang16 + dist10 + far1 + face8 + coin4 + see2 = 41
    obs_dim_onehot: int = 41
    # belief feature dim = 2(mean) + 1(mean_dist) + 1(var_trace) + 1(entropy) + 2*topk(=6) = 11
    belief_feat_dim: int = 11

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
    seq_len: int = 60        # burn_in + unroll

    eps_start: float = 0.2
    eps_final: float = 0.02
    eps_decay: float = 0.9995

    # viz pruning (시각화 사본 전용)
    viz_min_edge_count: int = 3
    viz_topk_per_node: int = 2
    viz_k_core: int = 2
    viz_keep_largest: bool = True

def obs_to_onehot(obs: Tuple[int,int,bool,int,int,int]) -> np.ndarray:
    ang, dist, far, face, coin_bin, see = obs
    v = []
    def oh(i, n):
        a = np.zeros(n, dtype=np.float32); a[int(i)%n] = 1.0; return a
    v.append(oh(ang, 16))                     # 16
    v.append(oh(9 if far else dist, 10))      # 10
    v.append(np.array([1.0 if far else 0.0], dtype=np.float32)) # 1
    v.append(oh(face, 8))                     # 8
    v.append(oh(coin_bin, 4))                 # 4
    v.append(oh(see, 2))                      # 2
    return np.concatenate(v, axis=0)          # 41

class ParticleFilter:
    def __init__(self, cfg: HybridConfig, grid_size:int, fov_radius:int, seed:int=0):
        self.cfg = cfg
        self.n = grid_size
        self.fov = fov_radius
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
        noise = self.cfg.pf_process_noise
        jitter = self.rng.normal(0.0, noise, size=self.p.shape).astype(np.float32)
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

        ang_sigma = self.cfg.pf_likelihood_ang_sigma
        dist_sigma = self.cfg.pf_likelihood_dist_sigma

        def ang_circ_delta(a,b):
            d = abs((a-b)%16)
            return min(d, 16-d)

        ang_err = np.array([ang_circ_delta(pred_ang[i], ang) for i in range(self.num)], dtype=np.float32)
        dist_err = np.abs(pred_dist - (9 if far else dist)).astype(np.float32)

        like_ang = np.exp(-(ang_err**2)/(2*(ang_sigma**2)))
        like_dst = np.exp(-(dist_err**2)/(2*(dist_sigma**2)))
        like = like_ang * like_dst + 1e-8

        self.w *= like
        s = float(self.w.sum(dtype=np.float64))
        if (not np.isfinite(s)) or (s <= 0.0):
            self.w.fill(1.0 / self.num)
        else:
            self.w = (self.w / s).astype(np.float32)

        if (self._steps % self.cfg.pf_resample_every)==0:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.num
        positions = (np.arange(N, dtype=np.float64) + float(self.rng.random())) / float(N)

        cumsum = np.cumsum(self.w, dtype=np.float64)
        if cumsum[-1] <= 0.0 or (not np.isfinite(cumsum[-1])):
            self.w.fill(1.0 / N)
            cumsum = np.cumsum(self.w, dtype=np.float64)
        cumsum[-1] = 1.0

        indexes = np.searchsorted(cumsum, positions, side='left')
        indexes = np.minimum(indexes, N - 1)

        self.p = self.p[indexes]
        self.w.fill(1.0 / N)

    def entropy(self) -> float:
        w = np.clip(self.w, 1e-12, 1.0)
        return float(-(w*np.log(w)).sum() / math.log(len(w)))

    def summarize(self, topk:int=None) -> np.ndarray:
        if topk is None: topk = self.cfg.pf_topk_modes
        mean = (self.w[:,None]*self.p).sum(axis=0)
        var_trace = (self.w[:,None]*((self.p-mean)**2)).sum(axis=0).sum()
        mean_dist = (np.abs(self.p).sum(axis=1)*self.w).sum()
        entropy = self.entropy()

        idx = np.argsort(-self.w)[:topk]
        top = self.p[idx]

        s = float(self.n)
        feats = [mean[0]/s, mean[1]/s, mean_dist/(2*s), min(1.0, var_trace/(s*s)), entropy]
        for i in range(topk):
            if i < len(top):
                feats.extend([float(top[i,0]/s), float(top[i,1]/s)])
            else:
                feats.extend([0.0, 0.0])
        # 총 11차원
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
    def __init__(self, capacity:int, seq_len:int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.data = []
        self.ptr = 0
    def push_episode(self, traj: Dict[str, np.ndarray]):
        if len(traj["act"]) < 2:
            return
        if len(self.data) < self.capacity:
            self.data.append(traj)
        else:
            self.data[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.capacity
    def sample_batch(self, batch:int, burn_in:int, unroll:int):
        if len(self.data) < batch: return None
        out = []
        need = burn_in + unroll + 1
        tries = 0
        while len(out) < batch and tries < 200:
            ep = random.choice(self.data)
            T = len(ep["act"])
            if T >= need:
                start = random.randint(0, T - need)
                sl = slice(start, start+need)
                out.append({k: v[sl] for k,v in ep.items()})
            tries += 1
        return out if len(out)==batch else None

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

    def train_batch(self, batch: List[Dict[str,np.ndarray]], gamma: float, n_step:int=1):
        if batch is None: return 0.0
        device = self.device
        B = len(batch)
        burn, unroll = self.cfg.burn_in, self.cfg.unroll

        def to_t(name):
            X = np.stack([b[name] for b in batch],0)
            return torch.from_numpy(X).float().to(device)

        x = to_t("x")                      # (B,T,dim)
        a = torch.from_numpy(np.stack([b["act"] for b in batch],0)).long().to(device)
        r = torch.from_numpy(np.stack([b["rew"] for b in batch],0)).float().to(device)
        d = torch.from_numpy(np.stack([b["done"] for b in batch],0)).float().to(device)

        # burn-in 은닉상태 추정
        _, h = self.net(x[:, :burn, :])

        # 온라인 Q (현재)
        q_on, _ = self.net(x[:, burn:-1, :], h)                      # (B, unroll, A)
        q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1)  # (B, unroll)

        # Double DQN target
        with torch.no_grad():
            q_on_next, _  = self.net(x[:, burn+1:, :], h)            # (B, unroll, A)
            a_star = torch.argmax(q_on_next, dim=-1)                 # (B, unroll)
            q_tgt_next, _ = self.tgt(x[:, burn+1:, :], h)            # (B, unroll, A)
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
        return float(loss.item())

# ============================ UTIL: movement delta ========================== #
def my_move_delta_from_action(action_idx:int, my_face_bin:int) -> Tuple[int,int]:
    if action_idx==6:   vx,vy = dir_to_vec(my_face_bin)         # FORWARD
    elif action_idx==7: vx,vy = dir_to_vec((my_face_bin-2)%8)   # STRAFE_L
    elif action_idx==8: vx,vy = dir_to_vec((my_face_bin+2)%8)   # STRAFE_R
    elif action_idx==9: vx,vy = dir_to_vec((my_face_bin+4)%8)   # BACKSTEP
    else:               vx,vy = (0,0)
    return (vx,vy)

# ============================ PROGRESS / FILTER ============================= #
def compute_pf_entropy(pf: ParticleFilter) -> float:
    return pf.entropy()

def is_informative_step(role:str,
                        obs_prev, obs_next,
                        r_who: float,
                        pf_ent_prev: float, pf_ent_new: float,
                        env_before: Tuple[Tuple[int,int],int,Tuple[int,int],int],
                        env_after:  Tuple[Tuple[int,int],int,Tuple[int,int],int],
                        coins_before:int, coins_after:int,
                        coin_dist_prev:int, coin_dist_new:int) -> bool:
    """
    POMDP 본질 유지: '관찰 변화/정보이득/목표진전' 중심의 필터.
    아래 중 하나라도 True면 informative:
      - 보상 발생 (|r| > eps)
      - far->near 전환 (LOS 진입/유지 변화)
      - PF 엔트로피 유의미 감소 (정보이득) : drop > 0.02
      - chaser: d_ce 감소, evader: 코인까지 거리 감소 또는 코인 수집 증가
    """
    eps = 1e-8
    if abs(r_who) > 1e-6:
        return True

    far_prev = obs_prev[2]; far_new = obs_next[2]
    if (far_prev and not far_new) or (not far_prev and far_new):
        return True

    if (pf_ent_prev - pf_ent_new) > 0.02:
        return True

    (c0, fc0, e0, fe0) = env_before
    (c1, fc1, e1, fe1) = env_after
    d0 = manhattan(c0, e0); d1 = manhattan(c1, e1)

    if role=="chaser":
        if d1 < d0: return True
    else:
        if coins_after > coins_before: return True
        if coin_dist_new < coin_dist_prev: return True

    return False

# ============================ RUN / LEARN (episode) ======================== #
def play_episode_and_learn(env: TwoAgentTagEnv,
                           who: str,
                           agent_self: HybridAgent,
                           agent_other: HybridAgent,
                           pf_self: ParticleFilter,
                           pf_oth: ParticleFilter,
                           replay: SeqReplay,
                           max_steps:int=1000):
    """한 에피소드 수행. 'who' 에이전트만 학습 데이터 수집/학습."""
    assert who in ("chaser","evader")
    env.reset()
    pf_self.reset(); pf_oth.reset()
    beliefs = BeliefIndexer()

    xs, acts, rews, dones = [], [], [], []
    h_self = None; h_oth = None

    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=False)
    ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=False)

    sum_r_c = 0.0; sum_r_e = 0.0
    steps = 0; captured=False

    # for info bonus & filter
    pf_ent_prev = compute_pf_entropy(pf_self)
    prev_evader_collected = env.evader_collected
    # evader coin distance (for filter)
    _, coin_d_prev = env._nearest_coin_dist(env.state[2])

    # 비정보 구간에서도 너무 희소해지지 않도록, N스텝마다 1프레임은 강제 보존
    KEEP_EVERY = 20
    keep_counter = 0

    while steps < max_steps:
        x_self = np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)
        x_oth  = np.concatenate([obs_to_onehot(ob_o), pf_oth.summarize()],  axis=0)

        a_s, h_self = agent_self.act_eps(x_self, h_self)
        a_o, h_oth  = agent_other.act_greedy(x_oth, h_oth)

        # env step (state snapshot for filter)
        state_before = env.state
        if who=="chaser":
            _, (r_c, r_e), done = env.step(a_s, a_o, eval_mode=False)
            r_who = r_c
        else:
            _, (r_c, r_e), done = env.step(a_o, a_s, eval_mode=False)
            r_who = r_e
        state_after = env.state

        sum_r_c += r_c; sum_r_e += r_e

        # PF 업데이트 (self)
        my_face_bin = ob_s[3]
        pf_self.predict(my_move=my_move_delta_from_action(a_s, my_face_bin))
        ob_s_next = env.observe(for_chaser=(who=="chaser"), eval_mode=False)
        pf_self.weight_update(ob_s_next)
        pf_ent_new = compute_pf_entropy(pf_self)

        # 정보이득 LOOK 보너스 (외부 shaping)
        if ACTIONS[a_s][0] == "LOOK":
            drop = pf_ent_prev - pf_ent_new
            if drop > 0.02:
                r_who += 0.0005  # k_info 역할(조건부)

        # 상대 PF도 갱신(평가·정책 결정을 위해)
        my_face_bin_o = ob_o[3]
        pf_oth.predict(my_move=my_move_delta_from_action(a_o, my_face_bin_o))
        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=False)
        pf_oth.weight_update(ob_o)

        # 필터 판단
        coin_d_new = env._nearest_coin_dist(env.state[2])[1]
        informative = is_informative_step(
            who, ob_s, ob_s_next, r_who,
            pf_ent_prev, pf_ent_new,
            state_before, state_after,
            prev_evader_collected, env.evader_collected,
            coin_d_prev, coin_d_new
        )
        keep_counter = (keep_counter + 1) % KEEP_EVERY

        if informative or keep_counter==0:
            xs.append(np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0))
            acts.append(a_s)
            rews.append(r_who)
            dones.append(1.0 if done else 0.0)

        # 다음 루프 준비
        ob_s = ob_s_next
        pf_ent_prev = pf_ent_new
        prev_evader_collected = env.evader_collected
        coin_d_prev = coin_d_new

        steps += 1
        if done:
            captured=True
            break

    # 최소 길이 가드
    if len(acts) < 2:
        # 마지막 상태라도 2스텝 맞추도록 보강
        if len(acts)==1:
            xs.append(xs[-1]); acts.append(acts[-1]); rews.append(rews[-1]); dones.append(1.0)
        else:
            # 전혀 없으면 더미 한 스텝 생성
            xs = [np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)]*2
            acts = [0,0]; rews=[0.0,0.0]; dones=[1.0,1.0]

    traj = {
        "x":   np.stack(xs,0).astype(np.float32),
        "act": np.array(acts, dtype=np.int64),
        "rew": np.array(rews, dtype=np.float32),
        "done":np.array(dones, dtype=np.float32)
    }
    replay.push_episode(traj)

    # 미니배치 학습
    batch = replay.sample_batch(agent_self.cfg.batch_size, agent_self.cfg.burn_in, agent_self.cfg.unroll)
    loss = agent_self.train_batch(batch, agent_self.cfg.gamma, n_step=agent_self.cfg.n_step) if batch else 0.0
    agent_self.decay_eps()

    return steps, sum_r_c, sum_r_e, captured, float(loss)

# ======================== EVAL: traj & bins (PNG용) ======================== #
def run_episode_collect_traj_hybrid(env, who: str,
                                    agent_self: HybridAgent,
                                    agent_other: HybridAgent,
                                    pf_self: ParticleFilter,
                                    pf_oth: ParticleFilter,
                                    max_steps:int=1000):
    env.reset()
    pf_self.reset(); pf_oth.reset()
    beliefs = BeliefIndexer()

    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
    ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
    b = beliefs.index(ob_s)
    traj_bins = [b]
    sum_r = 0.0
    actions_taken = []
    h_self = None; h_oth = None

    for _ in range(max_steps):
        x_self = np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)
        a_s, h_self = agent_self.act_greedy(x_self, h_self)

        x_oth = np.concatenate([obs_to_onehot(ob_o), pf_oth.summarize()], axis=0)
        a_o, h_oth = agent_other.act_greedy(x_oth, h_oth)

        if who=="chaser":
            _, (r_c, r_e), done = env.step(a_s, a_o, eval_mode=True)
            sum_r += r_c
        else:
            _, (r_c, r_e), done = env.step(a_o, a_s, eval_mode=True)
            sum_r += r_e

        my_face_bin = ob_s[3]
        pf_self.predict(my_move=my_move_delta_from_action(a_s, my_face_bin))
        ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
        pf_self.weight_update(ob_s)

        my_face_bin_o = ob_o[3]
        pf_oth.predict(my_move=my_move_delta_from_action(a_o, my_face_bin_o))
        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
        pf_oth.weight_update(ob_o)

        actions_taken.append(a_s)
        traj_bins.append(beliefs.index(ob_s))
        if done: break

    return {"sum_reward": sum_r,
            "steps": len(traj_bins)-1,
            "traj_bins": traj_bins,
            "actions": actions_taken}

# ======================== Episode Trajectory (PNG) ========================== #
def save_traj_png_only(traj_info, base_name: str, annotate_actions: bool = True):
    traj = traj_info["traj_bins"]
    acts = traj_info.get("actions", [])
    if not traj: return

    G = nx.DiGraph()
    unique_nodes = list(dict.fromkeys(traj))
    G.add_nodes_from(unique_nodes)
    for i in range(len(traj)-1):
        G.add_edge(traj[i], traj[i+1])

    first_visit = {}
    for step, b in enumerate(traj):
        if b not in first_visit: first_visit[b] = step

    max_step = max(first_visit.values()) if first_visit else 0
    denom = max(1, max_step)

    nodes = list(G.nodes())
    colors = [first_visit.get(n, 0)/denom for n in nodes]

    try:    pos = nx.kamada_kawai_layout(G)
    except: pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9,9))
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, arrows=False)

    sc = nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                node_size=25 if G.number_of_edges()>0 else 60,
                                node_color=colors, cmap=plt.cm.plasma,
                                vmin=0.0, vmax=1.0, alpha=0.95)

    if annotate_actions and len(traj)>1 and len(acts)>0:
        labels = {}
        for i in range(min(len(acts), len(traj)-1)):
            dst = traj[i+1]
            labels[dst] = ACTIONS[acts[i]][0]
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    if G.number_of_edges()==0:
        start_node = traj[0]
        nx.draw_networkx_labels(G, pos, labels={start_node:"START"}, font_size=7)

    plt.colorbar(sc, fraction=0.046, pad=0.04, label="first-visit (early→late)")
    plt.title(f"Episode diffusion (sum_r={traj_info['sum_reward']:.3f}, steps={traj_info['steps']})")
    plt.axis('off')
    png_path = os.path.join(MEDIA_DIR, f"{base_name}.png")
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close()
    print(f"[saved PNG] {png_path}")

    meta = {"base_name": base_name, "sum_reward": traj_info["sum_reward"],
            "steps": traj_info["steps"], "png": f"{base_name}.png"}
    with open(os.path.join(LOGS_DIR, f"{base_name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ======================= Combined Top-10% Graph (PNG) ====================== #
def build_combined_graph(top_eps: List[dict]):
    G = nx.DiGraph()
    node_visits = Counter()
    action_counter = defaultdict(Counter)
    for info in top_eps:
        traj = info["traj_bins"]; acts = info.get("actions", [])
        for b in traj: node_visits[b] += 1
        for i in range(len(traj)-1):
            u, v = traj[i], traj[i+1]
            G.add_edge(u, v)
            if i < len(acts):
                action_counter[v][acts[i]] += 1
    node_action_mode = {n: (action_counter[n].most_common(1)[0][0] if action_counter[n] else 0)
                        for n in G.nodes()}
    return G, node_action_mode, node_visits

def save_combined_graph_png(G, node_action_mode, node_visits,
                            base_name: str, title="Top-10% Combined Graph",
                            cap_edges: int = 12000):
    edges = list(G.edges())
    if len(edges) > cap_edges:
        rng = np.random.default_rng(42)
        edges = [edges[i] for i in rng.choice(len(edges), size=cap_edges, replace=False)]
    H = nx.DiGraph(); H.add_edges_from(edges)
    if H.number_of_nodes()>0:
        comps = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
        H = H.subgraph(comps[0]).copy()
    pos = nx.kamada_kawai_layout(H) if H.number_of_nodes()>0 else {}
    cmap = plt.cm.get_cmap("tab20", NUM_ACTIONS)
    nodes = list(H.nodes())
    colors = [cmap(node_action_mode.get(n, 0)) for n in nodes]
    sizes = [12 + 8*math.log2(1 + node_visits.get(n, 1)) for n in nodes]
    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(H, pos, alpha=0.12, width=0.6, arrows=False)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors, alpha=0.95)
    plt.title(title); plt.axis('off'); plt.tight_layout()
    png_path = os.path.join(MEDIA_DIR, f"{base_name}.png")
    plt.savefig(png_path, dpi=150); plt.close()
    print(f"[combined png] {png_path}")

# ============= Policy Graph (best-episode bins only) + PRUNING ============= #
@dataclass
class PolicyGraph:
    node_labels: Dict[int,int]
    edges: List[Tuple[int,int]]

def estimate_graph_hybrid(env: TwoAgentTagEnv,
                          beliefs: BeliefIndexer,
                          agent_self: HybridAgent,
                          agent_other: HybridAgent,
                          who: str,
                          bin_indices: List[int],
                          samples_per_bin:int=20,
                          rollout_len:int=2,
                          topk_per_node:int=2,
                          exclude_far_next: bool = True,
                          epsilon_eval: float = 0.10) -> PolicyGraph:
    node_labels: Dict[int,int] = {}
    transitions: Dict[int, Dict[int,int]] = {b:{} for b in bin_indices}
    action_mode: Dict[int, Counter] = {b:Counter() for b in bin_indices}

    def pick_eval_action(x, h):
        if random.random() < epsilon_eval:
            movable = [i for i,(k,_) in enumerate(ACTIONS)
                       if k in ("FORWARD","STRAFE_L","STRAFE_R","BACKSTEP",
                                "ROT+1","ROT-1","ROT+2","ROT-2","SPIN","TAKE_CORNER")]
            return random.choice(movable), h
        a, h2 = agent_self.act_greedy(x, h); return a, h2

    start_ts = time.strftime("%Y-%m-%d %H:%M:%S"); t0=time.time()
    print(f"[graph][start {start_ts}] {who}: bins={len(bin_indices)}, samples={samples_per_bin}, rollout={rollout_len}")

    pf = ParticleFilter(HybridConfig(), env.cfg.grid_size, env.cfg.fov_radius, seed=0)

    for b in bin_indices:
        for _ in range(samples_per_bin):
            env.reset(); pf.reset()
            for __ in range(24):
                ob = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
                if beliefs.index(ob)==b: break
                env.step(random.randrange(NUM_ACTIONS), random.randrange(NUM_ACTIONS), eval_mode=True)
            h_self=None; h_oth=None
            ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
            ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
            for __ in range(rollout_len):
                x = np.concatenate([obs_to_onehot(ob_s), pf.summarize()], axis=0)
                a_s, h_self = pick_eval_action(x, h_self)
                x_o = np.concatenate([obs_to_onehot(ob_o), pf.summarize()], axis=0)
                a_o, h_oth = agent_other.act_greedy(x_o, h_oth)
                if who=="chaser": env.step(a_s, a_o, eval_mode=True)
                else:             env.step(a_o, a_s, eval_mode=True)
                my_face_bin = ob_s[3]
                pf.predict(my_move=my_move_delta_from_action(a_s, my_face_bin))
                ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
                ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
                pf.weight_update(ob_s)

            nxt_obs = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
            if exclude_far_next and nxt_obs[2]:  # far=True
                continue
            nb = beliefs.index(nxt_obs)
            transitions[b][nb] = transitions[b].get(nb, 0) + 1
            action_mode[b][a_s] += 1

    edges = []
    for b in bin_indices:
        if not transitions[b]: continue
        items = sorted(transitions[b].items(), key=lambda kv: kv[1], reverse=True)
        for nb,_ in items[:topk_per_node]:
            if nb != b: edges.append((b,nb))
        node_labels[b] = action_mode[b].most_common(1)[0][0] if action_mode[b] else 0

    dur = time.time()-t0; end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[graph][end   {end_ts}] {who}: took {dur:.1f}s, edges={len(edges)}")
    return PolicyGraph(node_labels, edges)

def draw_policy_graph_pruned(pg: PolicyGraph,
                             title: str,
                             save_path: str,
                             k_core:int=2,
                             keep_largest:bool=True,
                             edge_cap:int=5000):
    nodes = set(pg.node_labels.keys())
    edges = [(u,v) for (u,v) in pg.edges if u in nodes and v in nodes and u!=v]
    if len(edges) > edge_cap:
        rng = np.random.default_rng(42)
        edges = [edges[i] for i in rng.choice(len(edges), size=edge_cap, replace=False)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes); G.add_edges_from(edges)
    if k_core and G.number_of_nodes()>0 and G.number_of_edges()>0:
        try: G = nx.k_core(G, k=k_core)
        except: pass
    if keep_largest and G.number_of_nodes()>0 and G.number_of_edges()>0:
        comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()

    if G.number_of_nodes()==0:
        plt.figure(figsize=(8,8)); plt.title(title+" (empty)")
        plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(); return

    pos = nx.kamada_kawai_layout(G)
    cols = [pg.node_labels.get(i, -1) for i in G.nodes()]
    plt.figure(figsize=(10,10)); plt.title(title)
    nx.draw_networkx_nodes(G, pos, node_size=28, node_color=cols, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.18, arrows=False, width=0.6)
    labels = {n: f"{pg.node_labels.get(n,0):02d}:{ACTIONS[pg.node_labels.get(n,0)][0]}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# =============================== IO HELPERS =================================#
class StopFlag:
    def __init__(self): self.flag=False
def stop_listener(sf: StopFlag):
    try:
        s = input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s == 'q': sf.flag=True
    except: pass

def save_model(applied_path_pt: str, meta_path_json: str, agent: HybridAgent, role: str):
    torch.save(agent.net.state_dict(), applied_path_pt)
    meta = {
        "role": role,
        "arch": "DRQN(Dueling)-GRU",
        "obs_dim": agent.cfg.obs_dim_onehot,
        "belief_dim": agent.cfg.belief_feat_dim,
        "hidden": agent.cfg.hidden,
        "n_actions": agent.cfg.n_actions,
        "gamma": agent.cfg.gamma,
        "eps": agent.eps,
        "step_count": agent.step_count,
        "file": os.path.basename(applied_path_pt)
    }
    with open(meta_path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# =========================== TRAIN FOREVER LOOP =============================#
def train_forever_accuracy(
    cfg: Config = Config(),
    SHOW_EVERY=100,        # 테스트에 적당. 크게 돌리려면 1000/10000
    PRINT_EVERY=10,
    traj_max_steps=1000,   # 에피소드 스텝가드 = 1000
    EVAL_EVERY=20,
    POOL_LIMIT=200
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TwoAgentTagEnv(cfg)
    beliefs = BeliefIndexer()
    hycfg = HybridConfig()

    chaser = HybridAgent(hycfg, device=device)
    evader = HybridAgent(hycfg, device=device)
    pf_ch = ParticleFilter(hycfg, cfg.grid_size, cfg.fov_radius, seed=0)
    pf_ev = ParticleFilter(hycfg, cfg.grid_size, cfg.fov_radius, seed=1)

    replay = SeqReplay(hycfg.replay_capacity, hycfg.seq_len)

    sf = StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    outer = 1
    while not sf.flag:
        # -------------------- Phase A: Chaser updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        best_score = -1e18; best_info = None
        ep_pool: List[dict] = []
        avg_loss = 0.0; loss_cnt=0

        for ep in range(1, SHOW_EVERY+1):
            if sf.flag: break
            steps, sr_c, sr_e, cap, loss = play_episode_and_learn(
                env, "chaser", chaser, evader, pf_ch, pf_ev, replay, max_steps=traj_max_steps)
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            avg_loss += loss; loss_cnt += 1
            if sr_c > best_score:
                best_score = sr_c
                best_info = run_episode_collect_traj_hybrid(env, "chaser", chaser, evader, pf_ch, pf_ev, max_steps=traj_max_steps)
            if (ep % PRINT_EVERY)==0:
                print(f"[Chaser] Outer {outer}  {ep}/{SHOW_EVERY}  eps={chaser.eps:.3f}  loss={ (avg_loss/max(1,loss_cnt)):.4f}")
            if (ep % EVAL_EVERY == 0) and (len(ep_pool) < POOL_LIMIT):
                info_eval = run_episode_collect_traj_hybrid(env, "chaser", chaser, evader, pf_ch, pf_ev, max_steps=traj_max_steps)
                ep_pool.append(info_eval)
        if sf.flag: break

        applied_pt = os.path.join(APPLIED_DIR, "chaser_policy.pt")
        applied_meta = os.path.join(APPLIED_DIR, "chaser_policy.meta.json")
        save_model(applied_pt, applied_meta, chaser, "chaser")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_pt_c = os.path.join(SNAPSHOT_DIR, f"chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.pt")
        torch.save(chaser.net.state_dict(), snap_pt_c)

        if best_info is not None and best_info["traj_bins"]:
            best_bins = sorted(set(best_info["traj_bins"]))
            pg_c = estimate_graph_hybrid(env, beliefs, chaser, evader, who="chaser",
                                         bin_indices=best_bins,
                                         samples_per_bin=20, rollout_len=2,
                                         topk_per_node=2, exclude_far_next=True, epsilon_eval=0.10)
            snap_png_c = os.path.join(SNAPSHOT_DIR, f"policy_graph_best_chaser_outer{outer}_{stamp}.png")
            draw_policy_graph_pruned(pg_c,
                                     f"Best-Episode Policy (Chaser, outer {outer})",
                                     snap_png_c, k_core=hycfg.viz_k_core,
                                     keep_largest=hycfg.viz_keep_largest, edge_cap=5000)

        chaser_log = {
            "phase": "chaser",
            "outer": outer,
            "episodes": SHOW_EVERY,
            "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY,
            "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY,
            "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.eps,
            "epsilon_evader": evader.eps,
            "avg_loss": (avg_loss/max(1,loss_cnt))
        }
        with open(os.path.join(LOGS_DIR, f"log_chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(chaser_log, f, ensure_ascii=False, indent=2)

        if best_info is not None:
            save_traj_png_only(best_info, base_name=f"best_chaser_outer{outer}_{stamp}", annotate_actions=True)

        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10 * len(ep_pool_sorted)))
            top_eps = ep_pool_sorted[:k]
            Gc, node_action_mode, node_visits = build_combined_graph(top_eps)
            save_combined_graph_png(Gc, node_action_mode, node_visits,
                                    base_name=f"combined_top10p_chaser_outer{outer}_{stamp}",
                                    title=f"Chaser Combined Top-10% (outer {outer})")

        # -------------------- Phase B: Evader updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        best_score = -1e18; best_info = None
        ep_pool = []; avg_loss=0.0; loss_cnt=0

        for ep in range(1, SHOW_EVERY+1):
            if sf.flag: break
            steps, sr_c, sr_e, cap, loss = play_episode_and_learn(
                env, "evader", evader, chaser, pf_ev, pf_ch, replay, max_steps=traj_max_steps)
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            avg_loss += loss; loss_cnt += 1
            if sr_e > best_score:
                best_score = sr_e
                best_info = run_episode_collect_traj_hybrid(env, "evader", evader, chaser, pf_ev, pf_ch, max_steps=traj_max_steps)
            if (ep % PRINT_EVERY)==0:
                print(f"[Evader] Outer {outer}  {ep}/{SHOW_EVERY}  eps={evader.eps:.3f}  loss={ (avg_loss/max(1,loss_cnt)):.4f}")
            if (ep % EVAL_EVERY == 0) and (len(ep_pool) < POOL_LIMIT):
                info_eval = run_episode_collect_traj_hybrid(env, "evader", evader, chaser, pf_ev, pf_ch, max_steps=traj_max_steps)
                ep_pool.append(info_eval)
        if sf.flag: break

        applied_pt = os.path.join(APPLIED_DIR, "evader_policy.pt")
        applied_meta = os.path.join(APPLIED_DIR, "evader_policy.meta.json")
        save_model(applied_pt, applied_meta, evader, "evader")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_pt_e = os.path.join(SNAPSHOT_DIR, f"evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.pt")
        torch.save(evader.net.state_dict(), snap_pt_e)

        if best_info is not None and best_info["traj_bins"]:
            best_bins = sorted(set(best_info["traj_bins"]))
            pg_e = estimate_graph_hybrid(env, beliefs, evader, chaser, who="evader",
                                         bin_indices=best_bins,
                                         samples_per_bin=20, rollout_len=2,
                                         topk_per_node=2, exclude_far_next=True, epsilon_eval=0.10)
            snap_png_e = os.path.join(SNAPSHOT_DIR, f"policy_graph_best_evader_outer{outer}_{stamp}.png")
            draw_policy_graph_pruned(pg_e,
                                     f"Best-Episode Policy (Evader, outer {outer})",
                                     snap_png_e, k_core=hycfg.viz_k_core,
                                     keep_largest=hycfg.viz_keep_largest, edge_cap=5000)

        evader_log = {
            "phase": "evader",
            "outer": outer,
            "episodes": SHOW_EVERY,
            "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY,
            "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY,
            "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.eps,
            "epsilon_evader": evader.eps,
            "avg_loss": (avg_loss/max(1,loss_cnt))
        }
        with open(os.path.join(LOGS_DIR, f"log_evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(evader_log, f, ensure_ascii=False, indent=2)

        if best_info is not None:
            save_traj_png_only(best_info, base_name=f"best_evader_outer{outer}_{stamp}", annotate_actions=True)

        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10 * len(ep_pool_sorted)))
            top_eps = ep_pool_sorted[:k]
            Ge, node_action_mode, node_visits = build_combined_graph(top_eps)
            save_combined_graph_png(Ge, node_action_mode, node_visits,
                                    base_name=f"combined_top10p_evader_outer{outer}_{stamp}",
                                    title=f"Evader Combined Top-10% (outer {outer})")

        outer += 1

# ================================ MAIN ===================================== #
if __name__ == "__main__":
    train_forever_accuracy(
        cfg=Config(),
        SHOW_EVERY=1000,      # 필요시 1000/10000으로
        PRINT_EVERY=100,
        traj_max_steps=1000, # 스텝가드 1000
        EVAL_EVERY=20,
        POOL_LIMIT=200
    )
