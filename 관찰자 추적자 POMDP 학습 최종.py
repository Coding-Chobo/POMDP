# -*- coding: utf-8 -*-
# Chaser–Evader POMDP (alternating updates, reward shaping)
# Checkpoint policy:
# - Every 10k episodes per phase (chaser/evader):
#   (1) applied JSON (overwrite)
#   (2) snapshot JSON (new file)
#   (3) policy graph PNG snapshot (new file) — LIGHT by default
#   (4) metrics LOG JSON (new file)
# - Heavy graph every 5th graph per agent (≈ 50k episodes per agent)
# - Graph estimation only on "active" bins for LIGHT (visits >= min_visits), to save time
# - Start/End timestamps for graph generation printed to console
# - Stop with 'q' + ENTER

import os, time, json, math, random, threading
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ============================ OUTPUT PATHS ================================= #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")  # 필요 시 절대경로로 변경 가능
APPLIED_DIR = os.path.join(OUTPUT_DIR, "applied")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(APPLIED_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================== CONFIG ==================================== #
@dataclass
class Config:
    grid_size: int = 30
    fov_radius: int = 10
    obs_noise: float = 0.12
    slip: float = 0.04
    capture_radius: int = 0
    seed: int = 42

    # Reward shaping (작게 시작)
    time_penalty: float = 0.0005
    k_distance: float  = 0.005
    k_los: float       = 0.002
    k_info: float      = 0.0005

    action_cost_wait: float   = 0.0
    action_cost_rot: float    = 0.0003
    action_cost_dash: float   = 0.0002
    action_cost_look: float   = 0.0003
    action_cost_strafe: float = 0.0001
    action_cost_back: float   = 0.0001
    action_cost_spin: float   = 0.0002

random.seed(42); np.random.seed(42)
def clamp(x, lo, hi): return max(lo, min(hi, x))
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def angle_bin_from_dxdy16(dx, dy):
    if dx == 0 and dy == 0: return 0
    ang = math.atan2(dy, dx);  ang = ang + 2*math.pi if ang < 0 else ang
    bin_size = 2*math.pi/16
    return int((ang + bin_size/2)//bin_size) % 16

DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def dir_to_vec(di): return DIRS8[di % 8]
def add_pos(p, delta, n): return (clamp(p[0]+delta[0],0,n-1), clamp(p[1]+delta[1],0,n-1))

# =============================== ACTIONS =================================== #
ACTIONS: List[Tuple[str, Tuple]] = [("WAIT", ())]
for d in range(8): ACTIONS.append(("MOVE_ABS", (d, 1)))
ACTIONS += [
    ("ROT", (+1,)), ("ROT", (-1,)), ("ROT", (+2,)), ("ROT", (-2,)),
    ("FACE_TARGET", ()), ("LOOK", ()),
    ("STRAFE", ("L", 1)), ("STRAFE", ("R", 1)),
    ("DASH", (2,)), ("BACKSTEP", (1,)), ("SPIN", ()),
]
NUM_ACTIONS = len(ACTIONS)
assert NUM_ACTIONS == 20

# =============================== ENV ======================================= #
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
            new_face = angle_bin_from_dxdy16(dx, dy) % 8
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
        prev_d = manhattan(c, e)

        if not eval_mode:
            if random.random() < self.cfg.slip: a_c = random.randrange(NUM_ACTIONS)
            if random.random() < self.cfg.slip: a_e = random.randrange(NUM_ACTIONS)

        c_next, fc_next = self._apply_action(c, fc, "chaser", a_c, e)
        e_next, fe_next = self._apply_action(e, fe, "evader", a_e, c)
        self.state = (c_next, fc_next, e_next, fe_next)

        done = manhattan(c_next, e_next) <= self.cfg.capture_radius

        # base rewards
        r_c = 1.0 if done else 0.0
        r_e = -1.0 if done else 0.01

        # shaping
        r_c -= self.cfg.time_penalty
        r_e -= self.cfg.time_penalty

        new_d  = manhattan(c_next, e_next)
        delta_d = prev_d - new_d
        r_c += self.cfg.k_distance * (delta_d)
        r_e += self.cfg.k_distance * (-delta_d)

        new_far = new_d > self.cfg.fov_radius
        if not new_far: r_c += self.cfg.k_los
        else:           r_e += self.cfg.k_los

        kind_c, _ = ACTIONS[a_c]; kind_e, _ = ACTIONS[a_e]
        if kind_c == "LOOK":        r_c += self.cfg.k_info
        if kind_c == "FACE_TARGET": r_c += 0.5 * self.cfg.k_info
        if kind_e == "LOOK":        r_e += self.cfg.k_info
        if kind_e == "FACE_TARGET": r_e += 0.5 * self.cfg.k_info

        def act_cost(kind: str) -> float:
            if   kind == "WAIT":    return self.cfg.action_cost_wait
            elif kind == "ROT":     return self.cfg.action_cost_rot
            elif kind == "DASH":    return self.cfg.action_cost_dash
            elif kind == "LOOK":    return self.cfg.action_cost_look
            elif kind == "STRAFE":  return self.cfg.action_cost_strafe
            elif kind == "BACKSTEP":return self.cfg.action_cost_back
            elif kind == "SPIN":    return self.cfg.action_cost_spin
            else:                   return 0.0
        r_c -= act_cost(kind_c)
        r_e -= act_cost(kind_e)

        if done:
            r_c = 1.0
            r_e = -1.0

        return self.state, (r_c, r_e), done

    def _dist_bin(self, d, cuts=(1,2,4,6,9,13,18,24)):
        for i,c in enumerate(cuts):
            if d <= c: return i
        return len(cuts)

    def observe(self, for_chaser: bool, eval_mode=False) -> Tuple[int,int,bool,int]:
        (c, fc, e, fe) = self.state
        me_pos, me_face = (c, fc) if for_chaser else (e, fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"
        bonus = self.look_bonus[key]
        eff_fov = int(self.cfg.fov_radius * (1.5 if (bonus and not eval_mode) else 1.0))
        eff_noise = 0.0 if eval_mode else self.cfg.obs_noise * (0.5 if bonus else 1.0)
        if not eval_mode:
            self.look_bonus[key] = False
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
        return (ang_bin, dist_bin, far, my_face_bin)

# ============================== BELIEF ===================================== #
class BeliefIndexer:
    def __init__(self, na=16, nd=10, nf=8):
        self.na, self.nd, self.nf = na, nd, nf
    def index(self, obs: Tuple[int,int,bool,int]) -> int:
        ang, dist, far, face = obs
        dcode = 9 if far else dist
        return ((face*self.nd) + dcode)*self.na + (ang % self.na)
    @property
    def n_bins(self): return self.na*self.nd*self.nf

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
    """
    방문횟수 N 합(행동 차원 합산)이 min_visits 이상인 belief-bin만 추립니다.
    너무 많으면 무작위로 max_bins개 샘플링.
    """
    Nsum = agent.N.sum(axis=1)  # shape: [n_bins]
    idx = np.where(Nsum >= float(min_visits))[0]
    if idx.size == 0:  # 학습 초반 대비
        idx = np.arange(agent.N.shape[0])
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
                   who: str, samples_per_bin=50, rollout_len=2,
                   bin_indices: Optional[List[int]] = None) -> PolicyGraph:
    n = beliefs.n_bins
    node_labels = {b:int(np.argmax((self_pol.Q1[b]+self_pol.Q2[b])*0.5)) for b in range(n)}

    if bin_indices is None:
        bin_indices = list(range(n))
    bin_set = set(bin_indices)

    transitions: Dict[int, Dict[int,int]] = {b:{} for b in bin_indices}

    start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    print(f"[graph][start {start_ts}] {who}: bins={len(bin_indices)}/{n}, "
          f"samples={samples_per_bin}, rollout={rollout_len}")

    for b in bin_indices:
        for _ in range(samples_per_bin):
            env.reset()
            # target bin 근처로 빠르게 이동시키기 위한 가벼운 시도
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
        if b in bin_set:
            if transitions[b]:
                nb = max(transitions[b].items(), key=lambda kv: kv[1])[0]
                edges.append((b, nb))
            else:
                edges.append((b, b))
        else:
            edges.append((b, b))

    dur = time.time()-t0
    end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[graph][end   {end_ts}] {who}: took {dur:.1f}s")
    return PolicyGraph(node_labels, edges)

def draw_policy_graph_nx(pg, beliefs, title, save_path, layout="kamada_kawai"):
    G = nx.DiGraph(); n = beliefs.n_bins
    G.add_nodes_from(range(n)); G.add_edges_from(pg.edges)
    node_color = [pg.node_labels.get(i,-1) for i in range(n)]
    # layout with graceful fallback
    try:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=1.2/np.sqrt(n))
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
    except Exception as e:
        print(f"[warn] layout='{layout}' failed ({type(e).__name__}: {e}). Falling back to kamada_kawai.")
        pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(11,11)); plt.title(title)
    nx.draw_networkx_nodes(G, pos, node_size=42, node_color=node_color, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=False, width=0.5)
    step = max(1, n//32); labels = {i:f"{i}\nA{pg.node_labels.get(i,-1)}" for i in range(0,n,step)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def print_graph_summary(pg, beliefs, k=12):
    lines=[]; step=max(1, beliefs.n_bins//k)
    for b in range(0, beliefs.n_bins, step):
        dst = next((v for (u,v) in pg.edges if u==b), b)
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

    # 각 에이전트의 그래프 생성 카운트 (헤비를 5회마다 실행하기 위함)
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

        # (1) 적용용 JSON 덮어쓰기
        save_agent_json(applied_chaser_path, chaser)

        # (2) 스냅샷 JSON
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_c = os.path.join(SNAPSHOT_DIR, f"chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_c, chaser)

        # (3) 그래프 (LIGHT 기본, 5번마다 HEAVY)
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
                             snap_png_c, nx_layout)
        print_graph_summary(pg_c, beliefs)

        # (4) 로그 기록용 JSON (지표 요약)
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
            "graph_mode": mode_tag,
            "graph_bins": "all" if bin_idx is None else int(len(bin_idx)),
            "graph_samples_per_bin": spb,
            "graph_rollout_len": rlen
        }
        log_path_c = os.path.join(LOGS_DIR, f"log_chaser_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        with open(log_path_c, "w", encoding="utf-8") as f: json.dump(chaser_log, f, ensure_ascii=False, indent=2)

        # -------------------- Phase B: Evader updates -------------------- #
        steps_acc = 0; r_c_acc = 0.0; r_e_acc = 0.0; cap_cnt = 0
        for ep in range(1, SHOW_EVERY + 1):
            if sf.flag: break
            steps, sr_c, sr_e, cap = q_episode_unbounded_double(env, beliefs, chaser, evader, update="evader")
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            if ep % PRINT_EVERY == 0:
                print(f"[Evader] Outer {outer}  {ep}/{SHOW_EVERY}")
        if sf.flag: break

        # (1) 적용용 JSON 덮어쓰기
        save_agent_json(applied_evader_path, evader)

        # (2) 스냅샷 JSON
        stamp = time.strftime("%Y%m%d-%H%M%S")
        snap_json_e = os.path.join(SNAPSHOT_DIR, f"evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        save_agent_json(snap_json_e, evader)

        # (3) 그래프 (LIGHT 기본, 5번마다 HEAVY)
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
                             snap_png_e, nx_layout)
        print_graph_summary(pg_e, beliefs)

        # (4) 로그 기록용 JSON (지표 요약)
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
            "graph_mode": mode_tag,
            "graph_bins": "all" if bin_idx is None else int(len(bin_idx)),
            "graph_samples_per_bin": spb,
            "graph_rollout_len": rlen
        }
        log_path_e = os.path.join(LOGS_DIR, f"log_evader_outer{outer}_ep{SHOW_EVERY}_{stamp}.json")
        with open(log_path_e, "w", encoding="utf-8") as f: json.dump(evader_log, f, ensure_ascii=False, indent=2)

        outer += 1

# ================================ MAIN ===================================== #
if __name__ == "__main__":
    train_forever_accuracy(
        cfg=Config(grid_size=30, fov_radius=10, obs_noise=0.12, slip=0.04, capture_radius=0),
        SHOW_EVERY=10_000,
        PRINT_EVERY=1_000,
        nx_layout="kamada_kawai",        # spring은 SciPy 필요할 수 있음
        # 라이트/헤비 파라미터 (원하면 여기서 조정)
        light_samples_per_bin=30,
        light_rollout_len=2,
        light_min_visits=5,
        light_max_bins=256,
        heavy_samples_per_bin=120,
        heavy_rollout_len=5
    )
