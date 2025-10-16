# run_pomdp_flee_infinite_logger.py
# 무제한 실행 + 20만 스텝 롤오버 + 시드 랜덤화 + NDJSON 로깅(현재 디렉토리 저장)
# 중단: Ctrl+C (KeyboardInterrupt)

import math
import random
import json
import time
import signal
from dataclasses import dataclass
from typing import Tuple, Dict, List
from pathlib import Path
from datetime import datetime

# =========================
# 1) 환경(Fleeing NPC)
# =========================

@dataclass(frozen=True)
class State:
    d: int   # 0: Near, 1: Mid, 2: Far
    vis: int # 0: Undetected, 1: Detected

class FleeEnv:
    ACTS = ["Hide","MoveAway","Wait","Decoy"]
    OBS  = ["StrongPing","WeakPing","NoPing"]

    def __init__(self, seed: int):
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

        # action costs
        if a == "Hide":       reward -= 0.5
        elif a == "MoveAway": reward -= 0.2
        elif a == "Decoy":    reward -= 0.3

        # risk penalty
        if s.vis == 1 and s.d == 0:
            reward -= 10.0

        d, vis = s.d, s.vis

        # transitions
        if a == "MoveAway" and self._r() < 0.8:
            d = min(2, d+1)
        if a == "Hide" and self._r() < 0.7:
            vis = 0
        if a == "Decoy" and self._r() < 0.5:
            vis = 0
        if a == "Wait":
            if vis == 1 and self._r() < 0.3: vis = 0
            if d == 0 and self._r() < 0.2: d = 1

        # small natural drift
        if d == 2 and vis == 1 and self._r() < 0.2:
            d = 1

        s_next = State(d, vis)

        # observation model
        if   (d,vis) == (0,1): probs = [0.75,0.20,0.05]
        elif (d,vis) == (0,0): probs = [0.40,0.45,0.15]
        elif (d,vis) == (1,1): probs = [0.50,0.35,0.15]
        elif (d,vis) == (1,0): probs = [0.20,0.50,0.30]
        elif (d,vis) == (2,1): probs = [0.25,0.45,0.30]
        else:                  probs = [0.05,0.25,0.70]  # Far & Undetected

        if a == "Decoy":
            probs = [max(0.0, probs[0]-0.1), min(1.0, probs[1]+0.1), probs[2]]
            ssum  = sum(probs); probs = [p/ssum for p in probs]

        # sample observation
        u = self._r(); acc = 0.0; o_idx = 2
        for i, p in enumerate(probs):
            acc += p
            if u <= acc:
                o_idx = i
                break

        return s_next, reward, o_idx, False, {}

    def _r(self) -> float:
        return self.rng.random()

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
    def __init__(self, env: FleeEnv, gamma=0.97, uct_c=1.0,
                 n_sims=800, max_depth=10, n_particles=500, seed=0):
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
        # rejection resampling to rebuild root particles
        new_particles, trials = [], 0
        limit = self.n_particles * 30
        while len(new_particles) < self.n_particles and trials < limit:
            trials += 1
            s_prev = self.env.sample_initial_state()
            s_next, _, obs_idx, _, _ = self.env.step(s_prev, a_idx)
            if obs_idx == o_idx:
                new_particles.append(s_next)
        if not new_particles:
            new_particles = [self.env.sample_initial_state() for _ in range(self.n_particles)]
        self.root.particles = new_particles

    # internals
    def _simulate(self, s: State, hist: Tuple[Tuple[int,int],...], depth: int) -> float:
        if depth >= self.max_depth:
            return 0.0
        bnode = self.tree.get(hist)
        if bnode is None:
            bnode = BeliefNode(); self.tree[hist] = bnode

        # expand lazily
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
# 3) NDJSON 로거
# =========================

class NDJSONLogger:
    """대용량 안전: 한 줄에 한 레코드(JSON)."""
    def __init__(self, path: Path, flush_every: int = 200):
        self.path = path
        self.f = path.open("a", encoding="utf-8")
        self.count = 0
        self.flush_every = flush_every

    def log_step(self, t: int, action: str, observation: str, reward: float,
                 hidden: State, particles: List[State], tree_nodes: int,
                 root_actions: Dict[int, ActionNode], env: FleeEnv):
        n = max(1, len(particles))
        p_near = sum(1 for s in particles if s.d == 0)/n
        p_det  = sum(1 for s in particles if s.vis == 1)/n

        counts = {}
        for s in particles:
            k = f"({s.d},{s.vis})"
            counts[k] = counts.get(k, 0) + 1

        ent = 0.0
        for c in counts.values():
            p = c / n
            ent -= p * math.log(p + 1e-12)

        root = { env.actions()[a_idx]: {"N": an.N, "Q": an.Q}
                 for a_idx, an in root_actions.items() }

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
# 4) 실행: 무제한 루프 + 20만 스텝 롤오버
# =========================

ROLL_STEPS = 200_000   # 파일 하나당 스텝 수 (요청대로 20만)
DEFAULT_N_SIMS = 800
DEFAULT_MAX_DEPTH = 10
DEFAULT_PARTICLES = 500

stop_flag = {"stop": False}
def _set_stop(signum, frame):
    stop_flag["stop"] = True
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _set_stop)

def new_run_file() -> Path:
    # 현재 디렉토리(실행 파일 위치/작업 디렉토리)에 저장
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path.cwd() / f"{stamp}.ndjson"

def run_once(steps: int, seed_env: int, seed_agent: int):
    env   = FleeEnv(seed=seed_env)
    agent = POMCP(env,
                  gamma=0.97, uct_c=1.0,
                  n_sims=DEFAULT_N_SIMS,
                  max_depth=DEFAULT_MAX_DEPTH,
                  n_particles=DEFAULT_PARTICLES,
                  seed=seed_agent)

    hidden = env.sample_initial_state()
    out = new_run_file()
    logger = NDJSONLogger(out, flush_every=500)
    print(f"[START] file={out.name}  steps={steps}  seeds(env={seed_env}, agent={seed_agent})")

    try:
        for t in range(steps):
            if stop_flag["stop"]:
                print("[STOP] received signal, finishing current file...")
                break

            a_idx = agent.plan()
            s_next, r, o_idx, done, _ = env.step(hidden, a_idx)

            logger.log_step(
                t=t,
                action=env.actions()[a_idx],
                observation=env.observations()[o_idx],
                reward=r,
                hidden=hidden,
                particles=agent.root.particles,
                tree_nodes=len(agent.tree),
                root_actions=agent.root.actions,
                env=env
            )

            agent.update_belief(a_idx, o_idx)
            hidden = s_next

            if (t+1) % 10_000 == 0:
                print(f"  t={t+1:,}  file={out.name}")

    finally:
        logger.close()
        print(f"[DONE] wrote {out.name}")

def main():
    print("[INFO] Infinite run. Press Ctrl+C to stop after current file is closed.")
    while not stop_flag["stop"]:
        # 에피소드마다 시드 랜덤화 (요청사항)
        seed_env = random.randint(0, 2**31 - 1)
        seed_agent = random.randint(0, 2**31 - 1)
        run_once(ROLL_STEPS, seed_env, seed_agent)
    print("[EXIT] bye.")

if __name__ == "__main__":
    main()
