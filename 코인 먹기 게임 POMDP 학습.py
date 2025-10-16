# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (coins + decoy + egocentric actions + alternating updates)
# - Coins: 5 items, each requires 5 COLLECT ticks (cumulative; no decay off-cell)
# - Evader DECOY: if chaser is within evader's FOV at use-time, success:
#       chaser slip += decoy_slip_add, chaser obs_noise *= decoy_noise_mult
#       effect lasts decoy_duration steps; evader gets r_decoy_success
# - Observation extended: + coin_dist_bin(4) + see_coin(2)  -> bins = 16*10*8*4*2 = 10240
# - Actions (unified for both agents; evader benefits from COLLECT/DECOY):
#   WAIT, ROT±1, ROT±2, SPIN, FORWARD, STRAFE_L, STRAFE_R, BACKSTEP, LOOK, FOCUS, TAKE_CORNER, COLLECT, DECOY
# - Alternating training forever; 10k episodes per phase; 'q'+ENTER to stop
# - Graph snapshots: LIGHT every 10k, HEAVY every 5th (≈50k); non-circular (kamada_kawai)

import os, time, json, math, random, threading
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless environment safe
import matplotlib.pyplot as plt
import networkx as nx

# ============================ OUTPUT PATHS ================================= #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
APPLIED_DIR = os.path.join(OUTPUT_DIR, "applied")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(APPLIED_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================== CONFIG ==================================== #
@dataclass
class Config:
    # World / observation
    grid_size: int = 20
    fov_radius: int = 10
    obs_noise: float = 0.12
    slip: float = 0.04
    capture_radius: int = 0     # 기본 0 (필요시 1로 바꿔 테스트 가능)
    seed: int = 42

    # Reward shaping
    time_penalty: float = 0.0005
    k_distance: float  = 0.005   # distance shaping (chaser +, evader -)
    k_los: float       = 0.002   # in-LOS bonus (chaser +, else evader +)
    k_info: float      = 0.0005  # LOOK/FACE-like info bonus (here only LOOK/FOCUS)

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
    r_coin_detect: float = 0.05        # coin newly enters FOV
    k_coin_approach: float = 0.002     # delta(nearest coin distance)
    r_coin_visible: float = 0.01       # coin visible in FOV
    r_duplicate_penalty: float = -0.05 # useless COLLECT (no coin here)

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
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def angle_bin_from_dxdy16(dx, dy):
    if dx == 0 and dy == 0: return 0
    ang = math.atan2(dy, dx)
    if ang < 0: ang += 2*math.pi
    bin_size = 2*math.pi/16
    return int((ang + bin_size/2)//bin_size) % 16

DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def dir_to_vec(di): return DIRS8[di % 8]
def add_pos(p, delta, n): return (clamp(p[0]+delta[0],0,n-1), clamp(p[1]+delta[1],0,n-1))

# =============================== ACTIONS =================================== #
# Egocentric moves only; no absolute move; FACE_TARGET removed; all step=1
# Unified action space for both agents (Evader benefits from COLLECT/DECOY)
ACTIONS: List[Tuple[str, Tuple]] = [
    ("WAIT", ()),
    ("ROT", (+1,)), ("ROT", (-1,)), ("ROT", (+2,)), ("ROT", (-2,)),
    ("SPIN", ()),
    ("FORWARD", (1,)),
    ("STRAFE", ("L", 1)), ("STRAFE", ("R", 1)),
    ("BACKSTEP", (1,)),
    ("LOOK", ()),
    ("FOCUS", ()),           # longer but weaker than LOOK (2 turns, 1.25x FOV, 0.5x noise)
    ("TAKE_CORNER", ()),     # rotate ±1 (choose shortest) + forward(1) in one action
    ("COLLECT", ()),         # meaningful for Evader (coin cell)
    ("DECOY", ()),           # meaningful for Evader
]
NUM_ACTIONS = len(ACTIONS)
assert NUM_ACTIONS == 15

# =============================== ENV ======================================= #
class TwoAgentTagEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser": 0, "evader": 0}   # turns left
        # coins
        self.coins = set()
        self.coin_ticks = {}     # {(x,y): accumulated ticks}
        self.evader_collected = 0
        self.coins_seen = set()  # positions seen at least once (for detect reward)
        # decoy
        self.decoy_timer = 0     # >0 means active effect on chaser
        self.training_role = None

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

        # place coins
        self.coins.clear(); self.coin_ticks.clear()
        forbid = {c, e}
        while len(self.coins) < self.cfg.coin_count:
            p = (random.randrange(self.n), random.randrange(self.n))
            if p not in forbid and p not in self.coins:
                self.coins.add(p); self.coin_ticks[p] = 0
        return self.state

    # helper: choose shortest ±1 rotation to face target pseudo (for TAKE_CORNER)
    def _delta_face_to(self, face_idx, dx, dy):
        # desired absolute 8-dir
        desired = angle_bin_from_dxdy16(dx, dy) % 8
        diff = (desired - face_idx) % 8
        if diff == 0: return 0
        # choose ±1 closest
        return +1 if diff <= 4 else -1

    def _apply_action(self, pos, face_idx, who: str, a_idx: int, other_pos):
        kind, param = ACTIONS[a_idx]
        n = self.n; new_pos, new_face = pos, face_idx

        if kind == "WAIT":
            pass

        elif kind == "ROT":
            step = param[0]; new_face = (new_face + step) % 8

        elif kind == "SPIN":
            new_face = (new_face + 4) % 8

        elif kind == "FORWARD":
            step = param[0]
            fdx, fdy = dir_to_vec(new_face)
            for _ in range(step): new_pos = add_pos(new_pos, (fdx,fdy), n)

        elif kind == "STRAFE":
            side, step = param; delta_face = -2 if side == "L" else +2
            sdx, sdy = dir_to_vec(new_face + delta_face)
            for _ in range(step): new_pos = add_pos(new_pos, (sdx,sdy), n)

        elif kind == "BACKSTEP":
            step = param[0]; bdx, bdy = dir_to_vec(new_face + 4)
            for _ in range(step): new_pos = add_pos(new_pos, (bdx,bdy), n)

        elif kind == "LOOK":
            # next 1 turn: FOV 1.5x, noise 0.5x
            self.look_bonus["chaser" if who=="chaser" else "evader"] = max(self.look_bonus["chaser" if who=="chaser" else "evader"], 1)

        elif kind == "FOCUS":
            # next 2 turns: FOV 1.25x, noise 0.5x
            turns = max(2, self.look_bonus["chaser" if who=="chaser" else "evader"])
            self.look_bonus["chaser" if who=="chaser" else "evader"] = turns

        elif kind == "TAKE_CORNER":
            # rotate ±1 (toward opponent) + forward(1)
            dx, dy = other_pos[0]-pos[0], other_pos[1]-pos[1]
            step = self._delta_face_to(new_face, dx, dy)
            new_face = (new_face + step) % 8
            fdx, fdy = dir_to_vec(new_face)
            new_pos = add_pos(new_pos, (fdx,fdy), n)

        elif kind in ("COLLECT", "DECOY"):
            # handled in step()
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

        # decay look/focus bonus
        if self.look_bonus["chaser"] > 0: self.look_bonus["chaser"] -= 1
        if self.look_bonus["evader"] > 0: self.look_bonus["evader"] -= 1
        # decoy timer
        if self.decoy_timer > 0: self.decoy_timer -= 1

        # slips (decoy affects chaser)
        slip_c = self.cfg.slip + (self.cfg.decoy_slip_add if self.decoy_timer > 0 else 0.0)
        slip_e = self.cfg.slip

        if not eval_mode:
            if random.random() < slip_c: a_c = random.randrange(NUM_ACTIONS)
            if random.random() < slip_e: a_e = random.randrange(NUM_ACTIONS)

        # apply movement & rotations
        c_next, fc_next = self._apply_action(c, fc, "chaser", a_c, e)
        e_next, fe_next = self._apply_action(e, fe, "evader", a_e, c)

        # ===== DECOY processing =====
        r_decoy = 0.0
        if ACTIONS[a_e][0] == "DECOY":
            # success if chaser within evader's FOV at the moment
            d_now = manhattan(e_next, c_next)
            if d_now <= self.cfg.fov_radius:
                self.decoy_timer = self.cfg.decoy_duration
                r_decoy = self.cfg.r_decoy_success

        # ===== COIN processing (cumulative 5 ticks; no off-cell decay) =====
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
                # pointless collect
                dup_penalty = self.cfg.r_duplicate_penalty

        # ===== State update =====
        self.state = (c_next, fc_next, e_next, fe_next)

        # ===== Terminal checks =====
        done_capture = manhattan(c_next, e_next) <= self.cfg.capture_radius
        done_allcoins = (self.evader_collected >= self.cfg.coin_count)
        done = done_capture or done_allcoins

        # ===== Base rewards =====
        r_c = 1.0 if done_capture else 0.0
        r_e = (-1.0 if done_capture else 0.01)

        # ===== Coin-aware shaping (detect/approach/visible) =====
        # compute coin features (before adding movement-based shaping)
        nearest_p, nearest_d_now = self._nearest_coin_dist(e_next)
        # detect: newly seen coin (in FOV) is added to coins_seen
        if nearest_p is not None and manhattan(e_next, nearest_p) <= self.cfg.fov_radius:
            if nearest_p not in self.coins_seen:
                r_e += self.cfg.r_coin_detect
                self.coins_seen.add(nearest_p)
            r_e += self.cfg.r_coin_visible

        # approach shaping: compare nearest distance before/after
        _, nearest_d_prev = self._nearest_coin_dist(e)
        if nearest_d_prev != 99 and nearest_d_now != 99:
            r_e += self.cfg.k_coin_approach * (nearest_d_prev - nearest_d_now)

        # add coin/decoy rewards
        r_e += r_collect_tick + r_collect_done + r_decoy + dup_penalty
        if done_allcoins: r_e += self.cfg.r_all_coins_bonus

        # ===== Generic shaping =====
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

        # Threat-aware: near chaser, discourage stubborn collect
        threat = (new_d <= self.cfg.threat_radius)
        if threat and kind_e not in ("COLLECT", "DECOY"):
            r_e += self.cfg.k_threat_avoid
        elif threat and kind_e == "COLLECT":
            r_e -= self.cfg.k_threat_avoid

        # ===== Action costs =====
        def act_cost(kind: str) -> float:
            if   kind == "WAIT":        return self.cfg.action_cost_wait
            elif kind == "ROT":         return self.cfg.action_cost_rot
            elif kind == "LOOK":        return self.cfg.action_cost_look
            elif kind == "FOCUS":       return self.cfg.action_cost_focus
            elif kind == "STRAFE":      return self.cfg.action_cost_strafe
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
        # 0: 0-2, 1:3-5, 2:6-9, 3:10+ / none
        if d <= 2: return 0
        if d <= 5: return 1
        if d <= 9: return 2
        return 3

    def observe(self, for_chaser: bool, eval_mode=False) -> Tuple[int,int,bool,int,int,int]:
        (c, fc, e, fe) = self.state
        me_pos, me_face = (c, fc) if for_chaser else (e, fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"

        # look/focus bonus
        bonus_turns = self.look_bonus[key]
        # evader's decoy affects only chaser's noise
        extra_noise_mult = (self.cfg.decoy_noise_mult if (for_chaser and self.decoy_timer>0) else 1.0)

        eff_fov = int(self.cfg.fov_radius * (1.5 if (bonus_turns >= 1 and not eval_mode) else (1.25 if (bonus_turns >= 2 and not eval_mode) else 1.0)))
        base_noise = self.cfg.obs_noise * (0.5 if (bonus_turns >= 1) else 1.0)
        eff_noise = 0.0 if eval_mode else base_noise * extra_noise_mult

        dx, dy = ot_pos[0]-me_pos[0], ot_pos[1]-me_pos[1]
        d = abs(dx)+abs(dy); far = d > eff_fov
        if far:
            ang_bin = 0; dist_bin = 9
        else:
            ang_bin = angle_bin_from_dxdy16(dx, dy)
            dist_bin = self._dist_bin(d)
            if random.random() < eff_noise: ang_bin = (ang_bin + random.choice([-1,1,2])) % 16
            if random.random() < eff_noise: dist_bin = clamp(dist_bin + random.choice([-1,1]), 0, 8)
        my_face_bin = me_face

        # coin features (for evader mainly, but unified)
        nearest_p, nearest_d = self._nearest_coin_dist((e if for_chaser else e))  # nearest coin wrt evader position
        coin_dist_bin = 3
        see_coin_flag = 0
        if nearest_p is not None:
            coin_dist_bin = self._coin_dist_bin(nearest_d)
            if nearest_d <= eff_fov:
                see_coin_flag = 1
                # coin observation noise (reuse eff_noise)
                if not eval_mode and random.random() < eff_noise:
                    see_coin_flag = 1 - see_coin_flag  # flip once

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
                               max_steps_guard=80_000, update: str = "both"):
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

# =================== ACTIVE BIN PICKER (for LIGHT graph) =================== #
def get_active_bins(agent: AgentDQ, min_visits=5, max_bins: Optional[int]=256, rng=None):
    Nsum = agent.N.sum(axis=1)
    idx = np.where(Nsum >= float(min_visits))[0]
    if idx.size == 0: idx = np.arange(agent.N.shape[0])
    if (max_bins is not None) and (idx.size > max_bins):
        if rng is None: rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=max_bins, replace=False)
    return np.sort(idx)

# ============================ POLICY GRAPH ================================= #
@dataclass
class PolicyGraph:
    node_labels: Dict[int,int]; edges: List[Tuple[int,int]]

def estimate_graph(env: TwoAgentTagEnv, beliefs: BeliefIndexer,
                   self_pol: AgentDQ, other_pol: AgentDQ,
                   who: str, samples_per_bin=40, rollout_len=2,
                   bin_indices: Optional[List[int]] = None) -> PolicyGraph:
    n = beliefs.n_bins
    node_labels = {b:int(np.argmax((self_pol.Q1[b]+self_pol.Q2[b])*0.5)) for b in range(n)}
    if bin_indices is None: bin_indices = list(range(n))
    bin_set = set(bin_indices)
    transitions: Dict[int, Dict[int,int]] = {b:{} for b in bin_indices}

    start_ts = time.strftime("%Y-%m-%d %H:%M:%S"); t0 = time.time()
    print(f"[graph][start {start_ts}] {who}: bins={len(bin_indices)}/{n}, samples={samples_per_bin}, rollout={rollout_len}")

    for b in bin_indices:
        for _ in range(samples_per_bin):
            env.reset()
            # cheap steering toward bin b
            for __ in range(48):
                ob = env.observe(who=="chaser", eval_mode=True)
                if beliefs.index(ob) == b: break
                env.step(0,0, eval_mode=True)
            for __ in range(rollout_len):
                ob_s = env.observe(who=="chaser", eval_mode=True)
                ob_o = env.observe(who!="chaser", eval_mode=True)
                bs = beliefs.index(ob_s); bo = beliefs.index(ob_o)
                a_s = self_pol.greedy_action(bs); a_o = other_pol.greedy_action(bo)
                env.step(a_s, a_o, eval_mode=True)
            nb = beliefs.index(env.observe(who=="chaser", eval_mode=True))
            transitions[b][nb] = transitions[b].get(nb, 0) + 1

    edges=[]
    for b in range(n):
        if b in bin_set and transitions[b]:
            nb = max(transitions[b].items(), key=lambda kv: kv[1])[0]
            if nb != b: edges.append((b, nb))  # remove self-loop
    dur = time.time()-t0; end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[graph][end   {end_ts}] {who}: took {dur:.1f}s")
    return PolicyGraph(node_labels, edges)

def draw_policy_graph_nx(pg, beliefs, title, save_path, layout="kamada_kawai", mode="light", bin_indices=None):
    n = beliefs.n_bins
    node_labels = pg.node_labels
    node_color = [node_labels.get(i,-1) for i in range(n)]
    nodes = set(range(n)) if bin_indices is None else set(bin_indices)
    edges = [(u,v) for (u,v) in pg.edges if u in nodes and v in nodes and u != v]

    # edge sampling for light mode (speed)
    if mode == "light" and len(edges) > 4000:
        rng = np.random.default_rng(42)
        edges = [edges[i] for i in rng.choice(len(edges), size=4000, replace=False)]

    G = nx.DiGraph()
    G.add_nodes_from(nodes); G.add_edges_from(edges)

    if G.number_of_edges() > 0 and G.number_of_nodes() > 0:
        comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    cols = [node_color[i] for i in G.nodes()]
    plt.figure(figsize=(10,10)); plt.title(title)
    nx.draw_networkx_nodes(G, pos, node_size=16 if mode=="light" else 38,
                           node_color=cols, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.12 if mode=="light" else 0.2, arrows=False, width=0.5)
    if mode != "light":
        step = max(1, n//40)
        labels = {i:f"{i}\nA{pg.node_labels.get(i,-1)}" for i in list(G.nodes())[::step]}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
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

# =========================== TRAIN FOREVER LOOP =============================#
def train_forever_accuracy(
    cfg: Config = Config(),
    SHOW_EVERY=10_000,
    PRINT_EVERY=1_000,
    nx_layout="kamada_kawai",
    # LIGHT defaults:
    light_samples_per_bin=30,
    light_rollout_len=2,
    light_min_visits=5,
    light_max_bins=256,
    # HEAVY defaults:
    heavy_samples_per_bin=120,
    heavy_rollout_len=5
):
    env = TwoAgentTagEnv(cfg)
    beliefs = BeliefIndexer()

    applied_chaser_path = os.path.join(APPLIED_DIR, "chaser_policy.json")
    applied_evader_path = os.path.join(APPLIED_DIR, "evader_policy.json")

    chaser = load_agent(applied_chaser_path, beliefs.n_bins, NUM_ACTIONS)
    evader = load_agent(applied_evader_path, beliefs.n_bins, NUM_ACTIONS)

    sf = StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    chaser_graph_count = 0
    evader_graph_count = 0
    outer = 1

    while not sf.flag:
        # -------------------- Phase A: Chaser updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        for ep in range(1, SHOW_EVERY + 1):
            if sf.flag: break
            steps, sr_c, sr_e, cap = q_episode_unbounded_double(env, beliefs, chaser, evader, update="chaser")
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            if ep % PRINT_EVERY == 0:
                print(f"[Chaser] Outer {outer}  {ep}/{SHOW_EVERY}")
        if sf.flag: break

        save_agent_json(applied_chaser_path, chaser)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_c = os.path.join(SNAPSHOT_DIR, f"chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_c, chaser)

        chaser_graph_count += 1
        is_heavy = (chaser_graph_count % 5 == 0)
        if is_heavy:
            bin_idx = None
            spb, rlen = heavy_samples_per_bin, heavy_rollout_len
            mode_tag = "heavy"
        else:
            bin_idx = get_active_bins(chaser, min_visits=light_min_visits, max_bins=light_max_bins)
            spb, rlen = light_samples_per_bin, light_rollout_len
            mode_tag = "light"

        print(f"[info] graph(chaser): mode={mode_tag}, count={chaser_graph_count}")
        pg_c = estimate_graph(env, beliefs, chaser, evader, who="chaser",
                              samples_per_bin=spb, rollout_len=rlen, bin_indices=bin_idx)
        snap_png_c = os.path.join(SNAPSHOT_DIR, f"policy_graph_chaser_outer{outer}_{mode_tag}_{stamp}.png")
        draw_policy_graph_nx(pg_c, beliefs,
                             f"Chaser Policy (outer {outer}, {mode_tag})",
                             snap_png_c, nx_layout, mode=mode_tag, bin_indices=bin_idx)
        print_graph_summary(pg_c, beliefs)

        chaser_log = {
            "phase": "chaser", "outer": outer, "episodes": SHOW_EVERY, "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY, "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY, "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.epsilon, "epsilon_evader": evader.epsilon,
            "graph_mode": mode_tag,
            "graph_bins": "all" if bin_idx is None else int(len(bin_idx)),
            "graph_samples_per_bin": spb, "graph_rollout_len": rlen
        }
        with open(os.path.join(LOGS_DIR, f"log_chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(chaser_log, f, ensure_ascii=False, indent=2)

        # -------------------- Phase B: Evader updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        for ep in range(1, SHOW_EVERY + 1):
            if sf.flag: break
            steps, sr_c, sr_e, cap = q_episode_unbounded_double(env, beliefs, chaser, evader, update="evader")
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            if ep % PRINT_EVERY == 0:
                print(f"[Evader] Outer {outer}  {ep}/{SHOW_EVERY}")
        if sf.flag: break

        save_agent_json(applied_evader_path, evader)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_e = os.path.join(SNAPSHOT_DIR, f"evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_e, evader)

        evader_graph_count += 1
        is_heavy = (evader_graph_count % 5 == 0)
        if is_heavy:
            bin_idx = None
            spb, rlen = heavy_samples_per_bin, heavy_rollout_len
            mode_tag = "heavy"
        else:
            bin_idx = get_active_bins(evader, min_visits=light_min_visits, max_bins=light_max_bins)
            spb, rlen = light_samples_per_bin, light_rollout_len
            mode_tag = "light"

        print(f"[info] graph(evader): mode={mode_tag}, count={evader_graph_count}")
        pg_e = estimate_graph(env, beliefs, evader, chaser, who="evader",
                              samples_per_bin=spb, rollout_len=rlen, bin_indices=bin_idx)
        snap_png_e = os.path.join(SNAPSHOT_DIR, f"policy_graph_evader_outer{outer}_{mode_tag}_{stamp}.png")
        draw_policy_graph_nx(pg_e, beliefs,
                             f"Evader Policy (outer {outer}, {mode_tag})",
                             snap_png_e, nx_layout, mode=mode_tag, bin_indices=bin_idx)
        print_graph_summary(pg_e, beliefs)

        evader_log = {
            "phase": "evader", "outer": outer, "episodes": SHOW_EVERY, "timestamp": stamp,
            "avg_steps": steps_acc/SHOW_EVERY, "avg_reward_chaser": r_c_acc/SHOW_EVERY,
            "avg_reward_evader": r_e_acc/SHOW_EVERY, "capture_rate": cap_cnt/SHOW_EVERY,
            "epsilon_chaser": chaser.epsilon, "epsilon_evader": evader.epsilon,
            "graph_mode": mode_tag,
            "graph_bins": "all" if bin_idx is None else int(len(bin_idx)),
            "graph_samples_per_bin": spb, "graph_rollout_len": rlen
        }
        with open(os.path.join(LOGS_DIR, f"log_evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(evader_log, f, ensure_ascii=False, indent=2)

        outer += 1

# ================================ MAIN ===================================== #
if __name__ == "__main__":
    train_forever_accuracy(
        cfg=Config(),  # capture_radius=0 유지 (필요시 Config(capture_radius=1))
        SHOW_EVERY=10_000,
        PRINT_EVERY=1_000,
        nx_layout="kamada_kawai",   # spring은 SciPy 의존 가능성 → 기본 kk
        light_samples_per_bin=30,
        light_rollout_len=2,
        light_min_visits=5,
        light_max_bins=256,
        heavy_samples_per_bin=150,
        heavy_rollout_len=6
    )
