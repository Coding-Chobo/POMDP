# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (Accuracy-focused)
# - Double Q-learning (tabular), per-(b,a) adaptive alpha, epsilon decay
# - Observation-binned belief with higher resolution (angle=16, distance bins refined, FAR)
# - Infinite alternating self-play; checkpoint every SHOW_EVERY episodes (JSON + NetworkX graph)
# - Policy graph estimation with noise-free eval, many samples, L-step rollout
# - Stop with 'q' + ENTER

import json, math, random, threading
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------- Config -------------------------------------- #
@dataclass
class Config:
    grid_size: int = 30
    fov_radius: int = 10
    obs_noise: float = 0.12
    slip: float = 0.04
    capture_radius: int = 0
    seed: int = 42

random.seed(42); np.random.seed(42)

def clamp(x, lo, hi): return max(lo, min(hi, x))
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

# 16 방향(22.5°)로 세분화 (정확도↑)
def angle_bin_from_dxdy16(dx, dy):
    if dx == 0 and dy == 0: return 0
    ang = math.atan2(dy, dx);  ang = ang + 2*math.pi if ang < 0 else ang
    bin_size = 2*math.pi/16
    return int((ang + bin_size/2)//bin_size) % 16

DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def dir_to_vec(di): return DIRS8[di % 8]
def add_pos(p, delta, n): return (clamp(p[0]+delta[0],0,n-1), clamp(p[1]+delta[1],0,n-1))

# ----------------------------- Actions (20) -------------------------------- #
ACTIONS: List[Tuple[str, Tuple]] = [("WAIT", ())]
for d in range(8): ACTIONS.append(("MOVE_ABS", (d, 1)))  # 1..8
ACTIONS += [
    ("ROT", (+1,)), ("ROT", (-1,)), ("ROT", (+2,)), ("ROT", (-2,)),
    ("FACE_TARGET", ()), ("LOOK", ()),
    ("STRAFE", ("L", 1)), ("STRAFE", ("R", 1)),
    ("DASH", (2,)), ("BACKSTEP", (1,)), ("SPIN", ()),
]
NUM_ACTIONS = len(ACTIONS)

# ----------------------------- Env ----------------------------------------- #
class TwoAgentTagEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser": False, "evader": False}
    def reset(self):
        while True:
            c = (random.randrange(self.n), random.randrange(self.n))
            e = (random.randrange(self.n), random.randrange(self.n))
            if manhattan(c,e) >= self.n//2: break
        self.state = (c, 0, e, 4)
        self.look_bonus = {"chaser": False, "evader": False}
        return self.state
    def _apply_action(self, pos, face_idx, who: str, a_idx: int, other_pos):
        kind, param = ACTIONS[a_idx]
        n = self.n; new_pos, new_face = pos, face_idx
        if kind == "WAIT":
            pass
        elif kind == "MOVE_ABS":
            d, step = param; dx, dy = DIRS8[d]
            for _ in range(step): new_pos = add_pos(new_pos, (dx,dy), n)
        elif kind == "ROT":
            step = param[0]; new_face = (new_face + step) % 8
        elif kind == "FACE_TARGET":
            dx, dy = other_pos[0]-pos[0], other_pos[1]-pos[1]
            new_face = angle_bin_from_dxdy16(dx, dy) % 8  # 시야 정렬(8방향으로 매핑)
        elif kind == "LOOK":
            self.look_bonus["chaser" if who=="chaser" else "evader"] = True
        elif kind == "STRAFE":
            side, step = param; delta_face = -2 if side == "L" else +2
            sdx, sdy = dir_to_vec(new_face + delta_face)
            for _ in range(step): new_pos = add_pos(new_pos, (sdx,sdy), n)
        elif kind == "DASH":
            step = param[0]; fdx, fdy = dir_to_vec(new_face)
            for _ in range(step): new_pos = add_pos(new_pos, (fdx,fdy), n)
        elif kind == "BACKSTEP":
            step = param[0]; bdx, bdy = dir_to_vec(new_face + 4)
            for _ in range(step): new_pos = add_pos(new_pos, (bdx,bdy), n)
        elif kind == "SPIN":
            new_face = (new_face + 4) % 8
        return new_pos, new_face
    def step(self, a_c: int, a_e: int, eval_mode=False):
        (c, fc, e, fe) = self.state
        if not eval_mode:
            if random.random() < self.cfg.slip: a_c = random.randrange(NUM_ACTIONS)
            if random.random() < self.cfg.slip: a_e = random.randrange(NUM_ACTIONS)
        c_next, fc_next = self._apply_action(c, fc, "chaser", a_c, e)
        e_next, fe_next = self._apply_action(e, fe, "evader", a_e, c)
        self.state = (c_next, fc_next, e_next, fe_next)
        done = manhattan(c_next, e_next) <= self.cfg.capture_radius
        r_c = 1.0 if done else 0.0; r_e = -1.0 if done else 0.01
        return self.state, (r_c, r_e), done
    def _dist_bin(self, d, cuts=(1,2,4,6,9,13,18,24)):  # 더 촘촘(정확도↑)
        for i,c in enumerate(cuts):
            if d <= c: return i
        return len(cuts)  # near bins 0..len(cuts) -> 0..8
    def observe(self, for_chaser: bool, eval_mode=False) -> Tuple[int,int,bool,int]:
        (c, fc, e, fe) = self.state
        me_pos, me_face = (c, fc) if for_chaser else (e, fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"
        bonus = self.look_bonus[key]
        eff_fov = int(self.cfg.fov_radius * (1.5 if (bonus and not eval_mode) else 1.0))
        eff_noise = 0.0 if eval_mode else self.cfg.obs_noise * (0.5 if bonus else 1.0)
        if not eval_mode:
            self.look_bonus[key] = False  # 보너스 소모(학습 시)
        dx, dy = ot_pos[0]-me_pos[0], ot_pos[1]-me_pos[1]
        d = abs(dx)+abs(dy); far = d > eff_fov
        if far:
            ang_bin = 0; dist_bin = 9  # FAR sentinel (near 0..8, FAR=9)
        else:
            ang_bin = angle_bin_from_dxdy16(dx, dy)  # 16방향
            dist_bin = self._dist_bin(d)            # 0..8
            if random.random() < eff_noise: ang_bin = (ang_bin + random.choice([-1,1,2])) % 16
            if random.random() < eff_noise: dist_bin = clamp(dist_bin + random.choice([-1,1]), 0, 8)
        my_face_bin = me_face  # 0..7
        return (ang_bin, dist_bin, far, my_face_bin)

# ----------------------- Belief (high-res) ---------------------------------- #
class BeliefIndexer:
    """
    bins = ang(16) * dist(10 incl FAR) * face(8) = 1280
    """
    def __init__(self, na=16, nd=10, nf=8):
        self.na, self.nd, self.nf = na, nd, nf
    def index(self, obs: Tuple[int,int,bool,int]) -> int:
        ang, dist, far, face = obs
        dcode = 9 if far else dist  # 0..8 near, 9 FAR
        return ((face*self.nd) + dcode)*self.na + (ang % self.na)
    @property
    def n_bins(self): return self.na*self.nd*self.nf

# -------------------- Double Q-learning (tabular) -------------------------- #
@dataclass
class AgentDQ:
    Q1: np.ndarray; Q2: np.ndarray
    N: np.ndarray   # visit count for adaptive alpha
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

def epsilon_schedule(e0=0.2, emin=0.02, decay=0.999995):
    e = e0
    while True:
        yield e
        e = max(emin, e*decay)

def q_episode_unbounded_double(env: TwoAgentTagEnv, beliefs: BeliefIndexer,
                               pol_c: AgentDQ, pol_e: AgentDQ,
                               max_steps_guard=80_000):
    env.reset(); steps = 0
    for pol in (pol_c, pol_e):  # 작은 탐색률 감소(스텝마다)
        pol.epsilon = max(0.02, pol.epsilon*0.999)

    while True:
        ob_c = env.observe(True); ob_e = env.observe(False)
        b_c = beliefs.index(ob_c); b_e = beliefs.index(ob_e)
        a_c = pol_c.act(b_c, True); a_e = pol_e.act(b_e, True)

        _, (r_c, r_e), done = env.step(a_c, a_e, eval_mode=False)

        nob_c = beliefs.index(env.observe(True))
        nob_e = beliefs.index(env.observe(False))

        # --- Double Q update (chaser) ---
        if random.random() < 0.5:
            # update Q1 using argmax from Q1, value from Q2
            a_star = int(np.argmax(pol_c.Q1[nob_c]))
            td = r_c + pol_c.gamma * pol_c.Q2[nob_c, a_star] - pol_c.Q1[b_c, a_c]
            pol_c.N[b_c, a_c] += 1.0; alpha = 1.0 / pol_c.N[b_c, a_c]
            pol_c.Q1[b_c, a_c] += alpha * td
        else:
            a_star = int(np.argmax(pol_c.Q2[nob_c]))
            td = r_c + pol_c.gamma * pol_c.Q1[nob_c, a_star] - pol_c.Q2[b_c, a_c]
            pol_c.N[b_c, a_c] += 1.0; alpha = 1.0 / pol_c.N[b_c, a_c]
            pol_c.Q2[b_c, a_c] += alpha * td

        # --- Double Q update (evader) ---
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
        if done or steps >= max_steps_guard: break

# ------------------------ Policy Graph (accurate) --------------------------- #
@dataclass
class PolicyGraph:
    node_labels: Dict[int,int]; edges: List[Tuple[int,int]]

def estimate_graph(env: TwoAgentTagEnv, beliefs: BeliefIndexer,
                   self_pol: AgentDQ, other_pol: AgentDQ,
                   who: str, samples_per_bin=150, rollout_len=5) -> PolicyGraph:
    n = beliefs.n_bins
    node_labels = {}
    for b in range(n):
        Qavg = (self_pol.Q1[b] + self_pol.Q2[b]) * 0.5
        node_labels[b] = int(np.argmax(Qavg))

    transitions: Dict[int, Dict[int,int]] = {b:{} for b in range(n)}
    for b in range(n):
        for _ in range(samples_per_bin):
            env.reset()
            # try to realize bin b (noise-free eval)
            for __ in range(96):
                ob = env.observe(who=="chaser", eval_mode=True)
                if beliefs.index(ob) == b: break
                env.step(0,0, eval_mode=True)

            # L-step rollout (greedy, eval_mode)
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
        if transitions[b]:
            nb = max(transitions[b].items(), key=lambda kv: kv[1])[0]
            edges.append((b, nb))
        else:
            edges.append((b, b))
    return PolicyGraph(node_labels, edges)

def draw_policy_graph_nx(pg, beliefs, title, save_path, layout="spring"):
    G = nx.DiGraph(); n = beliefs.n_bins
    G.add_nodes_from(range(n)); G.add_edges_from(pg.edges)
    node_color = [pg.node_labels.get(i,-1) for i in range(n)]
    pos = nx.spring_layout(G, seed=42, k=1.2/np.sqrt(n)) if layout=="spring" else nx.circular_layout(G)
    plt.figure(figsize=(11,11)); plt.title(title)
    nx.draw_networkx_nodes(G, pos, node_size=42, node_color=node_color, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=False, width=0.5)
    step = max(1, n//32); labels = {i:f"{i}\nA{pg.node_labels.get(i,-1)}" for i in range(0,n,step)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=140); plt.close()

def print_graph_summary(pg, beliefs, k=12):
    lines=[]; step=max(1, beliefs.n_bins//k)
    for b in range(0, beliefs.n_bins, step):
        dst = next((v for (u,v) in pg.edges if u==b), b)
        lines.append(f"[bin {b:04d}] greedy A={pg.node_labels.get(b,-1):02d}  --> {dst:04d}")
    print("\n=== Policy Graph (sample) ==="); print("\n".join(lines))

# --------------------------- Infinite Training ------------------------------ #
class StopFlag: 
    def __init__(self): self.flag=False
def stop_listener(sf: StopFlag):
    try:
        s = input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s == 'q': sf.flag=True
    except: pass

def save_agent(path, agent: AgentDQ):
    with open(path,"w",encoding="utf-8") as f: json.dump(agent.to_json(), f, ensure_ascii=False, indent=2)
def load_agent(path, bins, actions, gamma=0.997, epsilon=0.2):
    try:
        with open(path,"r",encoding="utf-8") as f: d = json.load(f)
        ag = AgentDQ.from_json(d)
        if ag.Q1.shape!=(bins,actions) or ag.Q2.shape!=(bins,actions): raise ValueError
        return ag
    except:
        return make_agent_dq(bins, actions, gamma, epsilon)

def train_forever_accuracy(
    cfg: Config = Config(),
    chaser_path="chaser_policy.json", evader_path="evader_policy.json",
    graph_dir=".", SHOW_EVERY=10_000, nx_layout="spring",
    samples_per_bin=150, rollout_len=5
):
    env = TwoAgentTagEnv(cfg); beliefs = BeliefIndexer()  # high-res bins
    chaser = load_agent(chaser_path, beliefs.n_bins, NUM_ACTIONS)
    evader = load_agent(evader_path, beliefs.n_bins, NUM_ACTIONS)

    sf = StopFlag(); threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()
    outer = 1
    while not sf.flag:
        # ---- Phase A: Chaser ----
        for _ in range(SHOW_EVERY):
            if sf.flag: break
            q_episode_unbounded_double(env, beliefs, chaser, evader)
        if sf.flag: break
        save_agent(chaser_path, chaser)
        pg_c = estimate_graph(env, beliefs, chaser, evader, who="chaser",
                              samples_per_bin=samples_per_bin, rollout_len=rollout_len)
        pth_c = f"{graph_dir}/policy_graph_chaser_outer{outer}_ep{SHOW_EVERY}.png"
        draw_policy_graph_nx(pg_c, beliefs, f"Chaser Policy (outer {outer}, +{SHOW_EVERY})", pth_c, nx_layout)
        print_graph_summary(pg_c, beliefs)

        # ---- Phase B: Evader ----
        for _ in range(SHOW_EVERY):
            if sf.flag: break
            q_episode_unbounded_double(env, beliefs, chaser, evader)
        if sf.flag: break
        save_agent(evader_path, evader)
        pg_e = estimate_graph(env, beliefs, evader, chaser, who="evader",
                              samples_per_bin=samples_per_bin, rollout_len=rollout_len)
        pth_e = f"{graph_dir}/policy_graph_evader_outer{outer}_ep{SHOW_EVERY}.png"
        draw_policy_graph_nx(pg_e, beliefs, f"Evader Policy (outer {outer}, +{SHOW_EVERY})", pth_e, nx_layout)
        print_graph_summary(pg_e, beliefs)
        outer += 1

if __name__ == "__main__":
    # 정확도 우선 기본값: samples_per_bin=150, rollout_len=5
    train_forever_accuracy(
        cfg=Config(grid_size=30, fov_radius=10, obs_noise=0.12, slip=0.04, capture_radius=0),
        chaser_path="chaser_policy.json",
        evader_path="evader_policy.json",
        graph_dir=".",
        SHOW_EVERY=10_000,
        nx_layout="spring",
        samples_per_bin=150,
        rollout_len=5
    )
