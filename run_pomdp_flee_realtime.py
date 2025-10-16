
# 목적: 도망 NPC POMDP + POMCP 온라인 계획을 실행하면서
#       매 step을 NDJSON(JSON Lines)로 로그 저장 (대용량에 안전).
# 사용: python run_pomdp_flee_logger.py
# 출력: ./runs/run_YYYYmmdd_HHMMSS.ndjson  (한 줄 = 한 스텝 레코드)

import math
import random
import json
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List
from pathlib import Path
from datetime import datetime

# =========================
# 1) 환경 (Fleeing NPC)
# =========================

@dataclass(frozen=True)
class State:
    d: int   # 0: Near, 1: Mid, 2: Far
    vis: int # 0: Undetected, 1: Detected

class FleeEnv:
    ACTS = ["Hide","MoveAway","Wait","Decoy"]
    OBS  = ["StrongPing","WeakPing","NoPing"]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def sample_initial_state(self) -> State:
        d = self.rng.choices([0,1,2], weights=[0.3,0.5,0.2])[0]
        vis = self.rng.choices([0,1], weights=[0.8,0.2])[0]
        return State(d, vis)

    def actions(self) -> List[str]:
        return self.ACTS

    def observations(self) -> List[str]:
        return self.OBS

    def step(self, s: State, a_idx: int) -> Tuple[State, float, int, bool, Dict]:
        a = self.ACTS[a_idx]
        reward = 1.0  # survival tick

        # 행동 비용
        if a == "Hide":     reward -= 0.5
        elif a == "MoveAway": reward -= 0.2
        elif a == "Decoy":  reward -= 0.3

        # 위험 패널티
        if s.vis == 1 and s.d == 0:
            reward -= 10.0

        d, vis = s.d, s.vis
        # 전이
        if a == "MoveAway" and self.rng.random() < 0.8:
            d = min(2, d+1)
        if a == "Hide" and self.rng.random() < 0.7:
            vis = 0
        if a == "Decoy" and self.rng.random() < 0.5:
            vis = 0
        if a == "Wait":
            if vis == 1 and self.rng.random() < 0.3: vis = 0
            if d == 0 and self.rng.random() < 0.2: d = 1
        # 자연 드리프트
        if d == 2 and vis == 1 and self.rng.random() < 0.2:
            d = 1

        s_next = State(d, vis)

        # 관찰모형
        if   (d,vis) == (0,1): probs = [0.75,0.20,0.05]
        elif (d,vis) == (0,0): probs = [0.40,0.45,0.15]
        elif (d,vis) == (1,1): probs = [0.50,0.35,0.15]
        elif (d,vis) == (1,0): probs = [0.20,0.50,0.30]
        elif (d,vis) == (2,1): probs = [0.25,0.45,0.30]
        else:                  probs = [0.05,0.25,0.70]  # Far & Undetected

        if a == "Decoy":
            # 강한 핑을 약하게 치우치게
            probs = [max(0.0, probs[0]-0.1), min(1.0, probs[1]+0.1), probs[2]]
            ssum  = sum(probs)
            probs = [p/ssum for p in probs]

        # 샘플링
        u = self.rng.random()
        acc = 0.0
        o_idx = 2
        for i,p in enumerate(probs):
            acc += p
            if u <= acc:
                o_idx = i
                break

        return s_next, reward, o_idx, False, {}

# =========================
# 2) POMCP (간단 구현)
# =========================

class ActionNode:
    __slots__ = ("N","Q","children")
    def __init__(self):
        self.N = 0
        self.Q = 0.0
        self.children: Dict[int, "BeliefNode"] = {}

class BeliefNode:
    __slots__ = ("N","actions","particles")
    def __init__(self):
        self.N = 0
        self.actions: Dict[int, ActionNode] = {}
        self.particles: List[State] = []

class POMCP:
    def __init__(self, env: FleeEnv, gamma=0.97, uct_c=1.0, n_sims=800, max_depth=10, n_particles=500, seed=11):
        self.env = env
        self.gamma = gamma
        self.uct_c = uct_c
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.n_particles = n_particles
        self.rng = random.Random(seed)

        self.root = BeliefNode()
        self.tree: Dict[Tuple[Tuple[int,int],...], BeliefNode] = {(): self.root}
        for _ in range(n_particles):
            self.root.particles.append(self.env.sample_initial_state())

    def plan(self) -> int:
        for _ in range(self.n_sims):
            if not self.root.particles:
                self.root.particles.append(self.env.sample_initial_state())
            s0 = self.rng.choice(self.root.particles)
            self._simulate(s0, (), 0)
        # argmax_a Q
        best_a, best_q = None, -1e100
        for a_idx, an in self.root.actions.items():
            if an.N > 0 and an.Q > best_q:
                best_a, best_q = a_idx, an.Q
        if best_a is None:
            best_a = self.rng.randrange(len(self.env.actions()))
        return best_a

    def update_belief(self, a_idx: int, o_idx: int):
        child = self._child_node((), a_idx, o_idx)
        self.root = child
        # 간단 리젝션 샘플링으로 루트 파티클 재구성
        new_particles, trials = [], 0
        while len(new_particles) < self.n_particles and trials < self.n_particles*30:
            trials += 1
            s_prev = self.env.sample_initial_state()
            s_next, _, obs_idx, _, _ = self.env.step(s_prev, a_idx)
            if obs_idx == o_idx:
                new_particles.append(s_next)
        if not new_particles:
            new_particles = [self.env.sample_initial_state() for _ in range(self.n_particles)]
        self.root.particles = new_particles

    # ---- internals ----
    def _simulate(self, s: State, hist: Tuple[Tuple[int,int],...], depth: int) -> float:
        if depth >= self.max_depth:
            return 0.0
        bnode = self.tree.get(hist)
        if bnode is None:
            bnode = BeliefNode(); self.tree[hist] = bnode

        if not bnode.actions:
            for a_idx in range(len(self.env.actions())):
                bnode.actions[a_idx] = ActionNode()
            return self._rollout(s, depth)

        total = bnode.N + 1e-8
        def uct(an: ActionNode) -> float:
            if an.N == 0: return float("inf")
            return an.Q + self.uct_c * math.sqrt(math.log(total)/an.N)

        a_idx, anode = max(bnode.actions.items(), key=lambda kv: uct(kv[1]))
        s_next, r, o_idx, done, _ = self.env.step(s, a_idx)
        child = self._child_node(hist, a_idx, o_idx)

        if done:
            G = r
        else:
            G = r + self.gamma * self._simulate(s_next, hist + ((a_idx,o_idx),), depth+1)

        # backup
        bnode.N += 1
        anode.N += 1
        anode.Q += (G - anode.Q) / anode.N
        return G

    def _rollout(self, s: State, depth: int) -> float:
        if depth >= self.max_depth:
            return 0.0
        a_idx = self.rng.randrange(len(self.env.actions()))
        s_next, r, o_idx, done, _ = self.env.step(s, a_idx)
        if done:
            return r
        return r + self.gamma * self._rollout(s_next, depth+1)

    def _child_node(self, hist: Tuple[Tuple[int,int],...], a_idx: int, o_idx: int) -> BeliefNode:
        bnode = self.tree.get(hist)
        if bnode is None:
            bnode = BeliefNode(); self.tree[hist] = bnode
        anode = bnode.actions.get(a_idx)
        if anode is None:
            anode = ActionNode(); bnode.actions[a_idx] = anode
        child_hist = hist + ((a_idx,o_idx),)
        child = self.tree.get(child_hist)
        if child is None:
            child = BeliefNode(); self.tree[child_hist] = child
        anode.children[o_idx] = child
        return child

# =========================
# 3) 로거 (NDJSON)
# =========================

class NDJSONLogger:
    """
    매 step마다 1줄(JSON) 기록. 대용량/긴 실행에 적합.
    각 줄 스키마 예:
      {
        "ts": 1739400000.123,           # epoch seconds
        "t": 42,                        # step
        "action": "Hide",
        "observation": "WeakPing",
        "reward": -0.2,
        "hidden_state": {"d":1,"vis":0},# 시뮬레이터 정답(학습용 supervised label)
        "belief": {
          "p_near": 0.31,
          "p_detected": 0.12,
          "counts": {"(0,0)":12,"(0,1)":3,"(1,0)":180,...},  # 파티클 분포
          "entropy": 1.73
        },
        "tree": {
          "nodes": 5231,
          "root_actions": {"Hide":{"N":850,"Q":3.1}, ...}
        }
      }
    """
    def __init__(self, out_path: Path, flush_every: int = 50):
        self.out_path = out_path
        self.f = out_path.open("a", encoding="utf-8")
        self.count = 0
        self.flush_every = flush_every

    def log_step(self, t: int, action: str, observation: str, reward: float,
                 hidden: State, particles: List[State], tree_nodes: int,
                 root_actions: Dict[int, ActionNode], env: FleeEnv):

        # belief summary
        n = max(1, len(particles))
        p_near = sum(1 for s in particles if s.d == 0)/n
        p_det  = sum(1 for s in particles if s.vis == 1)/n

        # detailed counts over (d,vis)
        counts = {}
        for s in particles:
            key = f"({s.d},{s.vis})"
            counts[key] = counts.get(key, 0) + 1

        # entropy over (d,vis) grid
        ent = 0.0
        for c in counts.values():
            p = c / n
            ent -= p * math.log(p + 1e-12)

        # root actions summary
        root = {}
        for a_idx, an in root_actions.items():
            root[env.actions()[a_idx]] = {"N": an.N, "Q": an.Q}

        rec = {
            "ts": time.time(),
            "t": t,
            "action": action,
            "observation": observation,
            "reward": reward,
            "hidden_state": {"d": hidden.d, "vis": hidden.vis},
            "belief": {"p_near": p_near, "p_detected": p_det,
                       "counts": counts, "entropy": ent},
            "tree": {"nodes": tree_nodes, "root_actions": root}
        }

        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

# =========================
# 4) 실행 루프
# =========================

def run_episode(T=200, seed_env=5, seed_agent=13,
                n_sims=800, max_depth=10, n_particles=500,
                log_dir: str = "runs"):

    env   = FleeEnv(seed=seed_env)
    agent = POMCP(env, gamma=0.97, uct_c=1.0,
                  n_sims=n_sims, max_depth=max_depth,
                  n_particles=n_particles, seed=seed_agent)

    hidden = env.sample_initial_state()

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_path = Path(log_dir) / f"{run_id}.ndjson"
    logger = NDJSONLogger(out_path)

    print(f"[RUN] writing NDJSON to: {out_path}")

    try:
        for t in range(T):
            # 계획
            a_idx = agent.plan()

            # 실제 환경 스텝
            s_next, r, o_idx, done, _ = env.step(hidden, a_idx)

            # 로그 (update_belief 전에, belief는 직전 루트)
            tree_nodes = len(agent.tree)
            root_actions = agent.root.actions
            logger.log_step(
                t=t,
                action=env.actions()[a_idx],
                observation=env.observations()[o_idx],
                reward=r,
                hidden=hidden,
                particles=agent.root.particles,
                tree_nodes=tree_nodes,
                root_actions=root_actions,
                env=env
            )

            # belief 업데이트
            agent.update_belief(a_idx, o_idx)

            # 다음 상태
            hidden = s_next

            # (선택) 콘솔 요약
            if t % 20 == 0:
                print(f" t={t:04d}  act={env.actions()[a_idx]:8s}  obs={env.observations()[o_idx]:10s}  "
                      f"reward={r:6.2f}  nodes={tree_nodes}")

        print(f"[DONE] steps={T}  file={out_path}")
    finally:
        logger.close()

if __name__ == "__main__":
    # 길게 돌릴수록 데이터가 축적됨. 필요시 T, n_sims 등을 늘리면 됨.
    run_episode(
        T=200_0000,        # 원하는 만큼 크게 (예: 100k도 가능, 파일만 커짐)
        seed_env=5,
        seed_agent=13,
        n_sims=800,
        max_depth=10,
        n_particles=500,
        log_dir="runs"
    )
