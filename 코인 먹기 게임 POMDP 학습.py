# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (coins + decoy + egocentric actions)
# - Alternating training forever (10k per phase)
# - Checkpoints every phase: applied JSON, snapshot JSON, policy graph (best-only), metrics log
# - Episode trajectory viz (PNG only; best-only)
#   * Single-episode PNG: first-visit-time color (diffusion look) + node action labels
#   * Combined Top-10% PNG: node color = dominant action, node size = visit frequency
# - Images saved under outputs/media/
# - JSON logs under outputs/logs/, applied policies under outputs/applied/
#
# NOTE (spec changes from previous version):
#   - Removed Light/Heavy modes (single best-episode policy graph per phase)
#   - Episode step guard set to 10,000
#   - Combined graph built from top 10% episodes (not fixed Top-100)
#   - Fixed manhattan() bug; made evader update symmetric (removed double apply)

import os, time, json, math, random, threading, shutil
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import networkx as nx

# ======================= FFMPEG AUTO-DETECT (PNG only but future-proof) ==== #
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
    capture_radius: int = 0     # 0=same cell only
    seed: int = 42

    # Reward shaping
    time_penalty: float = 0.0005
    k_distance: float  = 0.005
    k_los: float       = 0.002
    k_info: float      = 0.0005

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

    # Coin mission (5 items, 5 ticks each; cumulative)
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

random.seed(42); np.random.seed(42)
def clamp(x, lo, hi): return max(lo, min(hi, x))
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])  # fixed bug

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
# Egocentric action set (15): no absolute moves, no face-target
ACTIONS: List[Tuple[str, Tuple]] = [
    ("WAIT", ()),
    ("ROT+1", (+1,)), ("ROT-1", (-1,)), ("ROT+2", (+2,)), ("ROT-2", (-2,)),
    ("SPIN", ()),
    ("FORWARD", (1,)),
    ("STRAFE_L", ("L", 1)), ("STRAFE_R", ("R", 1)),
    ("BACKSTEP", (1,)),
    ("LOOK", ()),
    ("FOCUS", ()),           # 2 turns: FOV 1.25x, noise 0.5x
    ("TAKE_CORNER", ()),     # rotate ±1 toward opponent + forward(1)
    ("COLLECT", ()),         # Evader only (meaningful)
    ("DECOY", ()),           # Evader only (meaningful)
]
NUM_ACTIONS = len(ACTIONS)
assert NUM_ACTIONS == 15

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

        # COIN (cumulative, no off-cell decay)
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

# ============================== BELIEF ===================================== #
class BeliefIndexer:
    # (ang 16) * (dist 10 incl far) * (face 8) * (coin_dist 4) * (see_coin 2) = 10240
    def __init__(self, na=16, nd=10, nf=8, nc=4, ns=2):
        self.na, self.nd, self.nf, self.nc, self.ns = na, nd, nf, nc, ns
    def index(self, obs: Tuple[int,int,bool,int,int,int]) -> int:
        ang, dist, far, face, cbin, see = obs
        dcode = 9 if far else dist
        return (((((face*self.nd) + dcode)*self.na + (ang % self.na))*self.nc + cbin)*self.ns + see)
    @property
    def n_bins(self): return self.na*self.nd*self.nf*self.nc*self.ns  # 10240

# ================================ AGENT ==================================== #
@dataclass
class AgentDQ:
    Q1: np.ndarray; Q2: np.ndarray
    N: np.ndarray
    gamma: float; epsilon: float
    def act(self, b: int, train=True) -> int:
        if train and random.random() < self.epsilon: return random.randrange(self.Q1.shape[1])
        Qavg = (self.Q1[b] + self.Q2[b]) * 0.5
        return int(np.argmax(Qavg))
    def greedy_action(self, b: int) -> int:
        Qavg = (self.Q1[b] + self.Q2[b]) * 0.5
        return int(np.argmax(Qavg))
    def to_json(self):
        return {"gamma": self.gamma, "epsilon": self.epsilon,
                "Q1": self.Q1.tolist(), "Q2": self.Q2.tolist(), "N": self.N.tolist()}
    @staticmethod
    def from_json(d):
        return AgentDQ(Q1=np.array(d["Q1"], float), Q2=np.array(d["Q2"], float),
                       N=np.array(d["N"], float), gamma=float(d["gamma"]), epsilon=float(d["epsilon"]))

def make_agent_dq(bins: int, actions: int, gamma=0.997, epsilon=0.2):
    return AgentDQ(Q1=np.zeros((bins, actions), float),
                   Q2=np.zeros((bins, actions), float),
                   N=np.zeros((bins, actions), float),
                   gamma=gamma, epsilon=epsilon)

# 반환: steps, sum_r_c, sum_r_e, captured(bool)
def q_episode_unbounded_double(env: TwoAgentTagEnv, beliefs: BeliefIndexer,
                               pol_c: AgentDQ, pol_e: AgentDQ,
                               max_steps_guard=10_000, update: str = "both"):
    env.reset(); steps = 0
    sum_r_c = 0.0; sum_r_e = 0.0; captured = False
    for pol in (pol_c, pol_e):
        pol.epsilon = max(0.02, pol.epsilon * 0.999)

    while True:
        ob_c = env.observe(True); ob_e = env.observe(False)
        b_c = beliefs.index(ob_c); b_e = beliefs.index(ob_e)
        a_c = pol_c.act(b_c, True); a_e = pol_e.act(b_e, True)
        _, (r_c, r_e), done = env.step(a_c, a_e, eval_mode=False)
        sum_r_c += r_c; sum_r_e += r_e
        nob_c = beliefs.index(env.observe(True))
        nob_e = beliefs.index(env.observe(False))

        # --- Double Q-learning (symmetric) ---
        if update in ("both", "chaser"):
            if random.random() < 0.5:
                a_star = int(np.argmax(pol_c.Q1[nob_c]))
                td = r_c + pol_c.gamma * pol_c.Q2[nob_c, a_star] - pol_c.Q1[b_c, a_c]
                pol_c.N[b_c, a_c] += 1.0; alpha = 1.0 / pol_c.N[b_c, a_c]
                pol_c.Q1[b_c, a_c] += alpha * td
            else:
                a_star = int(np.argmax(pol_c.Q2[nob_c]))
                td = r_c + pol_c.gamma * pol_c.Q1[nob_c, a_star] - pol_c.Q2[b_c, a_c]
                pol_c.N[b_c, a_c] += 1.0; alpha = 1.0 / pol_c.N[b_c, a_c]
                pol_c.Q2[b_c, a_c] += alpha * td

        if update in ("both", "evader"):
            if random.random() < 0.5:
                a_star = int(np.argmax(pol_e.Q1[nob_e]))
                td = r_e + pol_e.gamma * pol_e.Q2[nob_e, a_star] - pol_e.Q1[b_e, a_e]
                pol_e.N[b_e, a_e] += 1.0; alpha = 1.0 / pol_e.N[b_e, a_e]
                pol_e.Q1[b_e, a_e] += alpha * td
            else:
                a_star = int(np.argmax(pol_e.Q2[nob_e]))
                td = r_e + pol_e.gamma * pol_e.Q1[nob_e, a_star] - pol_e.Q2[b_e, a_e]
                pol_e.N[b_e, a_e] += 1.0; alpha = 1.0 / pol_e.N[b_e, a_e]
                pol_e.Q2[b_e, a_e] += alpha * td

        steps += 1
        if done or steps >= max_steps_guard:
            captured = done
            break
    return steps, sum_r_c, sum_r_e, captured

# ============================ POLICY GRAPH ================================= #
@dataclass
class PolicyGraph:
    node_labels: Dict[int,int]; edges: List[Tuple[int,int]]

def estimate_graph(env: TwoAgentTagEnv, beliefs: BeliefIndexer,
                   self_pol: AgentDQ, other_pol: AgentDQ,
                   who: str, samples_per_bin=30, rollout_len=2,
                   bin_indices: Optional[List[int]] = None,
                   topk_per_node: int = 2,
                   exclude_far_next: bool = True,
                   epsilon_eval: float = 0.25) -> PolicyGraph:
    n = beliefs.n_bins
    node_labels = {b:int(np.argmax((self_pol.Q1[b]+self_pol.Q2[b])*0.5)) for b in range(n)}
    if bin_indices is None: bin_indices = list(range(n))
    transitions: Dict[int, Dict[int,int]] = {b:{} for b in bin_indices}
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S"); t0 = time.time()
    print(f"[graph][start {start_ts}] {who}: bins={len(bin_indices)}/{n}, samples={samples_per_bin}, rollout={rollout_len}")

    def pick_eval_action(agent, b):
        if random.random() < epsilon_eval:
            movable = [i for i,(k,_) in enumerate(ACTIONS)
                       if k in ("FORWARD","STRAFE_L","STRAFE_R","BACKSTEP","ROT+1","ROT-1","ROT+2","ROT-2","SPIN","TAKE_CORNER")]
            return random.choice(movable) if movable else agent.greedy_action(b)
        return agent.greedy_action(b)

    for b in bin_indices:
        for _ in range(samples_per_bin):
            env.reset()
            for __ in range(24):
                ob = env.observe(who=="chaser", eval_mode=True)
                if beliefs.index(ob) == b: break
                env.step(random.randrange(NUM_ACTIONS), random.randrange(NUM_ACTIONS), eval_mode=True)
            for __ in range(rollout_len):
                ob_s = env.observe(who=="chaser", eval_mode=True)
                ob_o = env.observe(who!="chaser", eval_mode=True)
                bs = beliefs.index(ob_s); bo = beliefs.index(ob_o)
                a_s = pick_eval_action(self_pol, bs); a_o = pick_eval_action(other_pol, bo)
                env.step(a_s, a_o, eval_mode=True)
            nxt_obs = env.observe(who=="chaser", eval_mode=True)
            if exclude_far_next and nxt_obs[2]:  # far=True
                continue
            nb = beliefs.index(nxt_obs)
            transitions[b][nb] = transitions[b].get(nb, 0) + 1

    edges=[]
    for b in bin_indices:
        if not transitions[b]: continue
        items = sorted(transitions[b].items(), key=lambda kv: kv[1], reverse=True)
        took = 0
        for nb,_ in items:
            if nb != b:
                edges.append((b, nb)); took += 1
                if took >= topk_per_node: break
    dur = time.time()-t0; end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[graph][end   {end_ts}] {who}: took {dur:.1f}s, edges={len(edges)}")
    return PolicyGraph(node_labels, edges)

def draw_policy_graph_nx(pg, beliefs, title, save_path, mode="light", bin_indices=None, edge_cap_light=5000):
    n = beliefs.n_bins
    node_labels = pg.node_labels
    nodes = set(range(n)) if bin_indices is None else set(bin_indices)
    edges = [(u,v) for (u,v) in pg.edges if u in nodes and v in nodes and u!=v]
    if mode=="light" and len(edges) > edge_cap_light:
        rng = np.random.default_rng(42)
        edges = [edges[i] for i in rng.choice(len(edges), size=edge_cap_light, replace=False)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes); G.add_edges_from(edges)
    if G.number_of_nodes()>0 and G.number_of_edges()>0:
        comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()
    pos = nx.kamada_kawai_layout(G) if G.number_of_nodes()>0 else {}
    cols = [node_labels.get(i,-1) for i in G.nodes()]
    plt.figure(figsize=(10,10)); plt.title(title)
    nx.draw_networkx_nodes(G, pos, node_size=18 if mode=="light" else 40,
                           node_color=cols, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.12 if mode=="light" else 0.18,
                           arrows=False, width=0.6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=140); plt.close()

def print_graph_summary(pg, beliefs, k=12):
    lines=[]; step=max(1, beliefs.n_bins//k)
    e_dict = {u:v for (u,v) in pg.edges}
    for b in range(0, beliefs.n_bins, step):
        dst = e_dict.get(b, b)
        lines.append(f"[bin {b:04d}] greedy A={pg.node_labels.get(b,-1):02d}  --> {dst:04d}")
    print("\n=== Policy Graph (sample) ==="); print("\n".join(lines))

# =============================== IO HELPERS =================================#
class StopFlag:
    def __init__(self): self.flag=False
def stop_listener(sf: StopFlag):
    try:
        s = input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s == 'q': sf.flag=True
    except: pass

def save_agent_json(path, agent: AgentDQ):
    with open(path,"w",encoding="utf-8") as f: json.dump(agent.to_json(), f, ensure_ascii=False, indent=2)

def load_agent(path, bins, actions, gamma=0.997, epsilon=0.2):
    try:
        with open(path,"r",encoding="utf-8") as f: d = json.load(f)
        ag = AgentDQ.from_json(d)
        if ag.Q1.shape!=(bins,actions) or ag.Q2.shape!=(bins,actions): raise ValueError
        return ag
    except:
        return make_agent_dq(bins, actions, gamma, epsilon)

# ======================== EPISODE TRAJECTORY (PNG 전용) ===================== #
def run_episode_collect_traj(env, beliefs, pol_self, pol_other, who: str,
                             max_steps=10_000, sample_frames=0):
    """
    평가 모드로 1에피소드를 실행하여 방문 bin 시퀀스를 기록.
    반환: dict {sum_reward, steps, traj_bins, actions}
    """
    env.reset()
    ob_s = env.observe(who=="chaser", eval_mode=True)
    b = beliefs.index(ob_s)
    traj_bins = [b]
    sum_r = 0.0
    actions_taken = []

    for _ in range(max_steps):
        if who == "chaser":
            a_s = pol_self.greedy_action(b)
            ob_o = env.observe(False, eval_mode=True)
            a_o = pol_other.greedy_action(beliefs.index(ob_o))
            _, (r_c, r_e), done = env.step(a_s, a_o, eval_mode=True)
            sum_r += r_c
            actions_taken.append(a_s)
        else:
            a_e = pol_self.greedy_action(b)
            ob_o = env.observe(True, eval_mode=True)
            a_c = pol_other.greedy_action(beliefs.index(ob_o))
            _, (r_c, r_e), done = env.step(a_c, a_e, eval_mode=True)
            sum_r += r_e
            actions_taken.append(a_e)

        ob_s = env.observe(who=="chaser", eval_mode=True)
        b = beliefs.index(ob_s)
        traj_bins.append(b)
        if done: break

    return {
        "sum_reward": sum_r,
        "steps": len(traj_bins)-1,
        "traj_bins": traj_bins,
        "actions": actions_taken
    }

def save_traj_png_only(traj_info, base_name: str, annotate_actions: bool = True):
    """
    순차 확산 느낌의 단일 PNG 시각화.
    - 전이가 0인 에피소드(노드 1개)도 안전하게 처리
    - 노드 색: 첫 방문 시점(정규화, 분모 최소 1로 가드)
    - 노드 라벨: 도달 시 취한 행동 이름
    """
    traj = traj_info["traj_bins"]
    acts = traj_info.get("actions", [])

    if not traj:
        print("[viz] empty traj; skip")
        return

    # 그래프 구성 (전이가 없어도 노드를 명시적으로 추가)
    G = nx.DiGraph()
    unique_nodes = list(dict.fromkeys(traj))  # 순서 보존 고유화
    G.add_nodes_from(unique_nodes)
    for i in range(len(traj) - 1):
        G.add_edge(traj[i], traj[i + 1])

    # 첫 방문 step 계산
    first_visit = {}
    for step, b in enumerate(traj):
        if b not in first_visit:
            first_visit[b] = step

    max_step = max(first_visit.values()) if first_visit else 0
    denom = max(1, max_step)  # ZeroDivision 방지

    nodes = list(G.nodes())
    colors = [first_visit.get(n, 0) / denom for n in nodes]  # 0~1 안전

    # 레이아웃
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9, 9))
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, arrows=False)

    sc = nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_size=25 if G.number_of_edges() > 0 else 60,
        node_color=colors,
        cmap=plt.cm.plasma,
        vmin=0.0, vmax=1.0,
        alpha=0.95
    )

    # 행동 라벨
    if annotate_actions and len(traj) > 1 and len(acts) > 0:
        labels = {}
        for i in range(min(len(acts), len(traj) - 1)):
            dst = traj[i + 1]
            labels[dst] = ACTIONS[acts[i]][0]
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    if G.number_of_edges() == 0:
        start_node = traj[0]
        nx.draw_networkx_labels(G, pos, labels={start_node: "START"}, font_size=7)

    plt.colorbar(sc, fraction=0.046, pad=0.04, label="first-visit (early→late)")
    plt.title(f"Episode diffusion (sum_r={traj_info['sum_reward']:.3f}, steps={traj_info['steps']})")
    plt.axis('off')
    png_path = os.path.join(MEDIA_DIR, f"{base_name}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[saved PNG] {png_path}")

    # 메타 JSON
    meta = {
        "base_name": base_name,
        "sum_reward": traj_info["sum_reward"],
        "steps": traj_info["steps"],
        "png": f"{base_name}.png"
    }
    with open(os.path.join(LOGS_DIR, f"{base_name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ======================= Top-10% 합성 그래프 (PNG) ========================= #
def build_combined_graph(top_eps: List[dict]):
    """
    top_eps: run_episode_collect_traj(...)로 얻은 dict들의 리스트 (traj_bins, actions 포함)
    반환: (G, node_action_mode, node_visits)
    """
    G = nx.DiGraph()
    node_visits = Counter()
    action_counter = defaultdict(Counter)  # node -> Counter(action)

    for info in top_eps:
        traj = info["traj_bins"]
        acts = info.get("actions", [])
        for b in traj:
            node_visits[b] += 1
        for i in range(len(traj)-1):
            u, v = traj[i], traj[i+1]
            G.add_edge(u, v)
            if i < len(acts):
                action_counter[v][acts[i]] += 1  # v 상태로 도달할 때 취한 행동

    node_action_mode = {}
    for n in G.nodes():
        node_action_mode[n] = action_counter[n].most_common(1)[0][0] if action_counter[n] else 0
    return G, node_action_mode, node_visits

def save_combined_graph_png(G, node_action_mode, node_visits,
                            base_name: str, title="Top-10% Combined Graph",
                            cap_edges: int = 12000):
    edges = list(G.edges())
    if len(edges) > cap_edges:
        rng = np.random.default_rng(42)
        edges = [edges[i] for i in rng.choice(len(edges), size=cap_edges, replace=False)]

    H = nx.DiGraph()
    H.add_edges_from(edges)
    if H.number_of_nodes() > 0:
        comps = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
        H = H.subgraph(comps[0]).copy()

    pos = nx.kamada_kawai_layout(H) if H.number_of_nodes() > 0 else {}
    cmap = plt.cm.get_cmap("tab20", NUM_ACTIONS)

    nodes = list(H.nodes())
    colors = [cmap(node_action_mode.get(n, 0)) for n in nodes]
    sizes = [12 + 8*math.log2(1 + node_visits.get(n, 1)) for n in nodes]  # log scale

    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(H, pos, alpha=0.12, width=0.6, arrows=False)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors, alpha=0.95)
    plt.title(title)
    plt.axis('off'); plt.tight_layout()
    png_path = os.path.join(MEDIA_DIR, f"{base_name}.png")
    plt.savefig(png_path, dpi=150); plt.close()
    print(f"[combined png] {png_path}")

# =========================== TRAIN FOREVER LOOP =============================#
def train_forever_accuracy(
    cfg: Config = Config(),
    SHOW_EVERY=10_000,
    PRINT_EVERY=1_000,
    # trajectory
    traj_max_steps=10_000,
    # pool/combined
    EVAL_EVERY=250,      # collect eval episodes into a pool
    POOL_LIMIT=500       # per phase
):
    env = TwoAgentTagEnv(cfg)
    beliefs = BeliefIndexer()

    applied_chaser_path = os.path.join(APPLIED_DIR, "chaser_policy.json")
    applied_evader_path = os.path.join(APPLIED_DIR, "evader_policy.json")
    chaser = load_agent(applied_chaser_path, beliefs.n_bins, NUM_ACTIONS)
    evader = load_agent(applied_evader_path, beliefs.n_bins, NUM_ACTIONS)

    sf = StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    outer = 1

    while not sf.flag:
        # -------------------- Phase A: Chaser updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        best_score = -1e18
        best_info = None
        ep_pool: List[dict] = []

        for ep in range(1, SHOW_EVERY + 1):
            if sf.flag: break
            steps, sr_c, sr_e, cap = q_episode_unbounded_double(env, beliefs, chaser, evader, update="chaser")
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            if sr_c > best_score:
                best_score = sr_c
                best_info = run_episode_collect_traj(env, beliefs, chaser, evader, "chaser",
                                                     max_steps=traj_max_steps, sample_frames=0)
            if (ep % PRINT_EVERY) == 0:
                print(f"[Chaser] Outer {outer}  {ep}/{SHOW_EVERY}")
            if (ep % EVAL_EVERY == 0) and (len(ep_pool) < POOL_LIMIT):
                info_eval = run_episode_collect_traj(env, beliefs, chaser, evader, "chaser",
                                                     max_steps=traj_max_steps, sample_frames=0)
                ep_pool.append(info_eval)
        if sf.flag: break

        # (1) 적용용 JSON 덮어쓰기
        save_agent_json(applied_chaser_path, chaser)

        # (2) 스냅샷 JSON
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_c = os.path.join(SNAPSHOT_DIR, f"chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_c, chaser)

        # (3) 최고 보상 에피소드 기반 정책 그래프 (방문 bin만)
        if best_info is not None:
            best_bins = sorted(set(best_info["traj_bins"]))
            pg_c = estimate_graph(env, beliefs, chaser, evader, who="chaser",
                                  samples_per_bin=30, rollout_len=2,
                                  bin_indices=best_bins,
                                  topk_per_node=2, exclude_far_next=True, epsilon_eval=0.25)
            snap_png_c = os.path.join(SNAPSHOT_DIR, f"policy_graph_best_chaser_outer{outer}_{stamp}.png")
            draw_policy_graph_nx(pg_c, beliefs, f"Best-Episode Policy (Chaser, outer {outer})",
                                 snap_png_c, mode="light", bin_indices=best_bins)
            print_graph_summary(pg_c, beliefs)

        # (4) 로그
        chaser_log = {
            "phase": "chaser",
            "outer": outer,
            "episodes": SHOW_EVERY,
            "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY,
            "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY,
            "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.epsilon,
            "epsilon_evader": evader.epsilon,
            "best_reward_in_phase": best_score
        }
        with open(os.path.join(LOGS_DIR, f"log_chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(chaser_log, f, ensure_ascii=False, indent=2)

        # --- Episode Trajectory Media (Best) — PNG only ---
        if best_info is not None:
            save_traj_png_only(best_info, base_name=f"best_chaser_outer{outer}_{stamp}",
                               annotate_actions=True)

        # 상위 10% 합성
        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10 * len(ep_pool_sorted)))   # top 10%
            top_eps = ep_pool_sorted[:k]
            Gc, node_action_mode, node_visits = build_combined_graph(top_eps)
            save_combined_graph_png(Gc, node_action_mode, node_visits,
                                    base_name=f"combined_top10p_chaser_outer{outer}_{stamp}",
                                    title=f"Chaser Combined Top-10% (outer {outer})")

        # -------------------- Phase B: Evader updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        best_score = -1e18
        best_info = None
        ep_pool = []

        for ep in range(1, SHOW_EVERY + 1):
            if sf.flag: break
            steps, sr_c, sr_e, cap = q_episode_unbounded_double(env, beliefs, chaser, evader, update="evader")
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            if sr_e > best_score:
                best_score = sr_e
                best_info = run_episode_collect_traj(env, beliefs, evader, chaser, "evader",
                                                     max_steps=traj_max_steps, sample_frames=0)
            if (ep % PRINT_EVERY) == 0:
                print(f"[Evader] Outer {outer}  {ep}/{SHOW_EVERY}")
            if (ep % EVAL_EVERY == 0) and (len(ep_pool) < POOL_LIMIT):
                info_eval = run_episode_collect_traj(env, beliefs, evader, chaser, "evader",
                                                     max_steps=traj_max_steps, sample_frames=0)
                ep_pool.append(info_eval)
        if sf.flag: break

        save_agent_json(applied_evader_path, evader)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_e = os.path.join(SNAPSHOT_DIR, f"evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_e, evader)

        if best_info is not None:
            best_bins = sorted(set(best_info["traj_bins"]))
            pg_e = estimate_graph(env, beliefs, evader, chaser, who="evader",
                                  samples_per_bin=30, rollout_len=2,
                                  bin_indices=best_bins,
                                  topk_per_node=2, exclude_far_next=True, epsilon_eval=0.25)
            snap_png_e = os.path.join(SNAPSHOT_DIR, f"policy_graph_best_evader_outer{outer}_{stamp}.png")
            draw_policy_graph_nx(pg_e, beliefs, f"Best-Episode Policy (Evader, outer {outer})",
                                 snap_png_e, mode="light", bin_indices=best_bins)
            print_graph_summary(pg_e, beliefs)

        evader_log = {
            "phase": "evader",
            "outer": outer,
            "episodes": SHOW_EVERY,
            "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY,
            "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY,
            "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.epsilon,
            "epsilon_evader": evader.epsilon,
            "best_reward_in_phase": best_score
        }
        with open(os.path.join(LOGS_DIR, f"log_evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(evader_log, f, ensure_ascii=False, indent=2)

        if best_info is not None:
            save_traj_png_only(best_info, base_name=f"best_evader_outer{outer}_{stamp}",
                               annotate_actions=True)

        if ep_pool:
            ep_pool_sorted = sorted(ep_pool, key=lambda d: d["sum_reward"], reverse=True)
            k = max(1, math.ceil(0.10 * len(ep_pool_sorted)))   # top 10%
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
        SHOW_EVERY=10_000,
        PRINT_EVERY=1_000,
        traj_max_steps=10_000,
        EVAL_EVERY=250,
        POOL_LIMIT=500
    )
