# hybrid_pf_lstm_pruning.py
# ------------------------------------------------------------
# 파티클 요약 + GRU(=LSTM 계열) 하이브리드 정책 + 시각화용 pruning
# ------------------------------------------------------------
import math, random, json, os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# 🔧 설정
# ==========================
@dataclass
class HybridConfig:
    grid_size: int = 20
    # Particle Filter
    pf_num_particles: int = 800
    pf_resample_every: int = 3
    pf_process_noise: float = 0.75    # 예측시 jitter 픽셀 표준편차
    pf_likelihood_ang_sigma: float = 1.0
    pf_likelihood_dist_sigma: float = 1.0
    pf_topk_modes: int = 3
    # DRQN / Training
    obs_dim_onehot: int = (16 + 10 + 1 + 8 + 4 + 2)  # ang16 + dist10 + far1 + face8 + coin4 + see2
    belief_feat_dim: int = 2 + 1 + 1 + 2*3  # mean(dx,dy)+ mean_dist + entropy + topK( (dx,dy)*K )
    hidden: int = 128
    n_actions: int = 15
    gamma: float = 0.997
    lr: float = 3e-4
    burn_in: int = 20
    unroll: int = 40
    target_update: int = 2000
    # Replay
    replay_capacity: int = 50_000
    batch_size: int = 16
    seq_len: int = 60          # burn_in+unroll과 동일하게 쓰는 걸 권장
    # Epsilon (행동 탐험)
    eps_start: float = 0.2
    eps_final: float = 0.02
    eps_decay: float = 0.9995
    # Viz & pruning (시각화 사본에만 적용)
    viz_min_edge_count: int = 3
    viz_topk_per_node: int = 2
    viz_k_core: int = 2
    viz_keep_largest: bool = True

# ============================================================
# 🧠 관찰 → 원-핫 벡터 (네 env.observe() 결과를 그대로 사용)
#   obs = (ang_bin, dist_bin, far(bool), my_face_bin, coin_dist_bin, see_coin_flag)
# ============================================================
def obs_to_onehot(obs: Tuple[int,int,bool,int,int,int]) -> np.ndarray:
    ang, dist, far, face, coin_bin, see = obs
    v = []
    def oh(i, n):
        a = np.zeros(n, dtype=np.float32); a[int(i)%n] = 1.0; return a
    v.append(oh(ang, 16))            # 16
    v.append(oh(9 if far else dist, 10))  # 10 (far면 dist=9 슬롯)
    v.append(np.array([1.0 if far else 0.0], dtype=np.float32))  # 1
    v.append(oh(face, 8))            # 8
    v.append(oh(coin_bin, 4))        # 4
    v.append(oh(see, 2))             # 2
    return np.concatenate(v, axis=0) # 총 41

# ============================================================
# 🎯 파티클 필터 (상대의 "상대좌표" (dx,dy)만 추적)
#   - 에이전트의 관점에서 상대까지의 상대좌표 분포를 입자로 유지
#   - 관찰 likelihood로 가중치 갱신
#   - 요약통계(평균/거리/엔트로피/Top-K 모드) 산출
# ============================================================
class ParticleFilter:
    def __init__(self, cfg: HybridConfig, fov_radius: int = 10, seed: int = 0):
        self.cfg = cfg
        self.n = cfg.grid_size
        self.fov = fov_radius
        self.rng = np.random.default_rng(seed)
        self.num = cfg.pf_num_particles
        self.p = None          # (N,2) dx,dy
        self.w = None          # (N,)
        self._steps = 0
        self.reset()

    def reset(self):
        # 초기 분포: 원점 주변 넓게 (상대좌표 모르므로 균일/가우시안)
        low = -self.n + 1; high = self.n - 1
        self.p = self.rng.integers(low, high+1, size=(self.num, 2)).astype(np.float32)
        self.w = np.ones(self.num, dtype=np.float32) / self.num
        self._steps = 0

    def predict(self, my_move: Tuple[int,int]=(0,0)):
        # 상대 좌표 = 상대 - 나
        # 내가 (dx,dy)로 움직였으면 상대좌표는 (-dx, -dy)만큼 변함
        dx, dy = my_move
        self.p[:,0] -= dx
        self.p[:,1] -= dy
        # 상대의 미지 이동(약한 랜덤 워크)
        noise = self.cfg.pf_process_noise
        jitter = self.rng.normal(0.0, noise, size=self.p.shape).astype(np.float32)
        self.p += jitter
        # 경계 클램프
        self.p[:,0] = np.clip(self.p[:,0], -self.n+1, self.n-1)
        self.p[:,1] = np.clip(self.p[:,1], -self.n+1, self.n-1)
        self._steps += 1

    def _angle_bin16(self, dx, dy):
        if dx==0 and dy==0: return 0
        a = math.atan2(dy, dx)
        if a<0: a += 2*math.pi
        bin_size = 2*math.pi/16
        return int((a + bin_size/2)//bin_size) % 16

    def _dist_bin10(self, d):
        # 네 코드의 dist_bin과 9=far를 맞추려면 far는 likelihood에서 따로 처리
        cuts = [1,2,4,6,9,13,18,24]  # 0..8 (9는 far)
        for i,c in enumerate(cuts):
            if d<=c: return i
        return len(cuts) # 8

    def weight_update(self, obs: Tuple[int,int,bool,int,int,int]):
        ang, dist, far, face, coin_bin, see = obs
        # 파티클의 관찰 예측치 계산
        dx = self.p[:,0]; dy = self.p[:,1]
        dist_mh = np.abs(dx) + np.abs(dy)
        pred_far = dist_mh > self.fov
        pred_ang = np.array([self._angle_bin16(dx[i], dy[i]) if not pred_far[i] else 0
                             for i in range(self.num)], dtype=np.int32)
        pred_dist = np.array([9 if pred_far[i] else self._dist_bin10(dist_mh[i])
                              for i in range(self.num)], dtype=np.int32)

        # 단순 가우시안 유사도(각도는 순환거리, 거리bin은 L2)
        ang_sigma = self.cfg.pf_likelihood_ang_sigma
        dist_sigma = self.cfg.pf_likelihood_dist_sigma

        def ang_circ_delta(a,b):  # 16개 원형 거리
            d = abs((a-b)%16)
            return min(d, 16-d)

        ang_err = np.array([ang_circ_delta(pred_ang[i], ang) for i in range(self.num)], dtype=np.float32)
        dist_err = np.abs(pred_dist - (9 if far else dist)).astype(np.float32)

        like_ang = np.exp(-(ang_err**2)/(2*(ang_sigma**2)))
        like_dst = np.exp(-(dist_err**2)/(2*(dist_sigma**2)))

        like = like_ang * like_dst + 1e-8
        self.w *= like
        s = self.w.sum()
        if s<=0:
            # 숫자 불안정 시 재초기화(희귀): 균일 재시작
            self.w[:] = 1.0/self.num
        else:
            self.w /= s

        # 주기적으로 리샘플
        if (self._steps % self.cfg.pf_resample_every) == 0:
            self._systematic_resample()

    def _systematic_resample(self):
        N = self.num
        positions = (np.arange(N) + self.rng.random())/N
        cumsum = np.cumsum(self.w)
        indexes = np.zeros(N, dtype=np.int32)
        i = j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.p = self.p[indexes]
        self.w = np.ones(N, dtype=np.float32)/N

    def summarize(self, topk:int=None) -> np.ndarray:
        if topk is None: topk = self.cfg.pf_topk_modes
        # 평균/분산/엔트로피
        mean = (self.w[:,None]*self.p).sum(axis=0)   # (2,)
        var = (self.w[:,None]*((self.p-mean)**2)).sum(axis=0).sum()  # trace approx
        mean_dist = (np.abs(self.p).sum(axis=1)*self.w).sum()  # E[manhattan]
        # 엔트로피(가중치 분포)
        w = np.clip(self.w, 1e-12, 1.0)
        entropy = float(-(w*np.log(w)).sum() / math.log(len(w)))  # [0,1] 정규화 근사

        # Top-K 모드(가중치 상위 K 파티클 위치)
        idx = np.argsort(-self.w)[:topk]
        top = self.p[idx]  # (K,2)

        # 스케일 정규화(대략적인 [-1,1])
        s = float(self.n)
        feats = [
            mean[0]/s, mean[1]/s,
            mean_dist/(2*s),            # manhattan 거리 정규화
            min(1.0, var/(s*s)),        # 대충 스케일
            entropy
        ]
        for i in range(topk):
            dx,dy = (top[i]/s) if i < len(top) else (np.array([0.0,0.0]))
            feats.extend([float(dx), float(dy)])
        # 부족한 K는 0으로 채워서 고정길이
        while len(feats) < (2 + 1 + 1 + 2*topk):  # mean2 + mean_dist1 + var1 + (dx,dy)*K
            feats.extend([0.0, 0.0])
        return np.array(feats, dtype=np.float32)

# ============================================================
# 🧩 DRQN(여기서는 GRU) + Dueling Head
#   입력 = 관찰 원-핫 + 파티클 요약
# ============================================================
class DRQN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.gru = nn.GRU(128, hidden, batch_first=True)
        # Dueling
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x_seq: torch.Tensor, h0: Optional[torch.Tensor]=None):
        # x_seq: [B,T,in_dim]
        z = self.enc(x_seq)
        out, hT = self.gru(z, h0)   # out: [B,T,H]
        V = self.val(out)           # [B,T,1]
        A = self.adv(out)           # [B,T,A]
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hT

# ============================================================
# 🧳 시퀀스 리플레이 버퍼 (간단 구현: 에피소드 단위로 저장)
# ============================================================
class SeqReplay:
    def __init__(self, capacity:int, seq_len:int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.data = []  # 각 항목은 dict{obs, belief, act, rew, done}
        self.ptr = 0

    def push_episode(self, traj: Dict[str, np.ndarray]):
        # traj 각 키: (T, feat) 혹은 (T,)
        if len(self.data) < self.capacity:
            self.data.append(traj)
        else:
            self.data[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch:int, burn_in:int, unroll:int):
        # 에피소드에서 랜덤 시작 위치를 뽑아 [burn_in+unroll] 길이로 슬라이스
        B = batch
        seqs = random.sample(self.data, B)
        out = []
        for ep in seqs:
            T = len(ep["act"])
            if T < (burn_in+unroll+1):   # 타깃 s_{t+unroll}까지 필요
                # 짧으면 패스 (실전엔 패딩/마스킹 구현 권장)
                return None
            start = random.randint(0, T - (burn_in+unroll+1))
            sl = slice(start, start+burn_in+unroll+1)  # +1 for bootstrap state
            item = {k: v[sl] for k,v in ep.items()}
            out.append(item)
        return out

# ============================================================
# 🧪 ε-greedy 행동 선택
# ============================================================
def select_action_eps_greedy(q_t: torch.Tensor, eps: float) -> int:
    # q_t: [1,1,A]
    if random.random() < eps:
        return random.randrange(q_t.shape[-1])
    return int(torch.argmax(q_t, dim=-1).item())

# ============================================================
# 🔍 시각화 + 프루닝 (사본에만 적용)
# ============================================================
def prune_transitions_by_count_and_topk(G: nx.DiGraph,
                                        min_edge_count:int=3,
                                        topk_per_node:int=2) -> nx.DiGraph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    # 엣지 가중치는 'count' 속성으로 기대
    for u in G.nodes():
        nbrs = []
        for v in G.successors(u):
            c = G[u][v].get("count", 1)
            if c >= min_edge_count and u!=v:
                nbrs.append((v,c))
        nbrs.sort(key=lambda x: x[1], reverse=True)
        for v,c in nbrs[:topk_per_node]:
            H.add_edge(u, v, count=c)
    return H

def prune_k_core_and_largest(G: nx.DiGraph, k_core:int=2, keep_largest:bool=True) -> nx.DiGraph:
    if G.number_of_nodes()==0: return G.copy()
    H = G.copy()
    if k_core is not None and k_core>0:
        try:
            H = nx.k_core(H, k=k_core)
        except nx.NetworkXError:
            pass
    if keep_largest and H.number_of_nodes()>0 and H.number_of_edges()>0:
        comps = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
        H = H.subgraph(comps[0]).copy()
    return H

def draw_graph_pruned_copy(G: nx.DiGraph, save_path:str,
                           min_edge_count:int=3, topk_per_node:int=2,
                           k_core:int=2, keep_largest:bool=True,
                           title:str="Policy graph (pruned copy)"):
    # 1) 원본 보존
    H = prune_transitions_by_count_and_topk(G, min_edge_count, topk_per_node)
    H = prune_k_core_and_largest(H, k_core, keep_largest)

    # 2) 레이아웃/그림
    if H.number_of_nodes()==0:
        plt.figure(figsize=(8,8)); plt.title(title+" (empty)")
        plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(); return

    pos = nx.kamada_kawai_layout(H)
    counts = [H[u][v].get("count",1) for u,v in H.edges()]
    nx.draw_networkx_edges(H, pos, alpha=0.18, arrows=False, width=[0.4+0.05*c for c in counts])
    nx.draw_networkx_nodes(H, pos, node_size=28, node_color="tab:blue", alpha=0.9)
    # (선택) 라벨: 너무 복잡해지면 생략 가능
    # nx.draw_networkx_labels(H, pos, font_size=6)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    return H

# ============================================================
# 🔌 통합 포인트 (예시 러너) — 네 env를 여기에 연결
#   - env.observe(True/False) : 관찰 튜플
#   - env.step(a_chaser, a_evader, eval_mode=False) : 한 스텝 진행
#   - ACTIONS : 행동 목록 (길이 = cfg.n_actions)
#   - 아래 함수 두 개만 네 프로젝트에 맞춰 채워 넣으면 끝!
# ============================================================
def my_move_delta_from_action(action_idx:int, my_face_bin:int) -> Tuple[int,int]:
    """내가 이 행동을 했을 때 상대좌표가 얼마나 바뀌는가를 '내 움직임' 기준으로 근사.
    - 네 프로젝트에서 정확한 이동(전진/스트레프/백스텝)을 계산할 수 있으면 여기를 치환하면 된다.
    - 간단 예시: 전진(1) = 내 페이스 방향으로 (dx,dy)=vec, 스트레프/백스텝은 그에 맞춰."""
    # 예시(대충): FORWARD=6, STRAFE_L=7, STRAFE_R=8, BACKSTEP=9 (네 ACTIONS 인덱스에 맞게 수정!)
    FORWARD, STRAFE_L, STRAFE_R, BACKSTEP = 6, 7, 8, 9
    # 8방향 단위 벡터
    DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    def vec(f): return DIRS8[f%8]
    dx,dy = 0,0
    if action_idx == FORWARD:
        vx,vy = vec(my_face_bin); dx,dy = vx,vy
    elif action_idx == STRAFE_L:
        vx,vy = vec((my_face_bin-2)%8); dx,dy = vx,vy
    elif action_idx == STRAFE_R:
        vx,vy = vec((my_face_bin+2)%8); dx,dy = vx,vy
    elif action_idx == BACKSTEP:
        vx,vy = vec((my_face_bin+4)%8); dx,dy = vx,vy
    else:
        dx,dy = 0,0
    return (dx,dy)

class HybridAgent:
    """파티클 요약 + GRU 기반 DRQN 정책 (ε-greedy)"""
    def __init__(self, cfg: HybridConfig, device="cpu"):
        self.cfg = cfg
        in_dim = cfg.obs_dim_onehot + cfg.belief_feat_dim
        self.net = DRQN(in_dim, cfg.hidden, cfg.n_actions).to(device)
        self.tgt = DRQN(in_dim, cfg.hidden, cfg.n_actions).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.device = device
        self.eps = cfg.eps_start
        self.step_count = 0

    def act(self, obs_vec: np.ndarray, belief_feat: np.ndarray, h: Optional[torch.Tensor]) -> Tuple[int, torch.Tensor]:
        x = torch.from_numpy(np.concatenate([obs_vec, belief_feat],axis=0)).float().to(self.device)
        x = x.view(1,1,-1)
        with torch.no_grad():
            q, h2 = self.net(x, h)
        a = select_action_eps_greedy(q, self.eps)
        return a, h2

    def decay_eps(self):
        self.eps = max(self.cfg.eps_final, self.eps*self.cfg.eps_decay)

    def train_on_batch(self, batch: List[Dict[str,np.ndarray]], gamma:float):
        if batch is None: return 0.0
        device = self.device
        B = len(batch); T = len(batch[0]["act"])  # T = burn+unroll+1
        burn, unroll = self.cfg.burn_in, self.cfg.unroll

        # (B,T,input_dim)
        def to_torch(name):
            X = np.stack([b[name] for b in batch], axis=0)  # (B,T,dim)
            return torch.from_numpy(X).float().to(device)

        x = to_torch("x")             # obs+belief
        a = torch.from_numpy(np.stack([b["act"] for b in batch],0)).long().to(device)  # (B,T)
        r = torch.from_numpy(np.stack([b["rew"] for b in batch],0)).float().to(device) # (B,T)
        d = torch.from_numpy(np.stack([b["done"] for b in batch],0)).float().to(device)# (B,T)

        # 1) burn-in: 상태만 데우기
        with torch.no_grad():
            _, h = self.net(x[:, :burn, :])

        # 2) unroll: 손실 계산
        q_on, _ = self.net(x[:, burn:-1, :], h)       # (B, unroll, A)
        with torch.no_grad():
            q_tgt, _ = self.tgt(x[:, burn+1:, :], h)  # 다음 상태 (B, unroll, A)

        # Double DQN target
        with torch.no_grad():
            q_online_next, _ = self.net(x[:, burn+1:, :], h)
            a_star = torch.argmax(q_online_next, dim=-1)                 # (B,unroll)
            q_next = q_tgt.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # (B,unroll)
            y = r[:, burn:-1] + gamma * (1.0 - d[:, burn:-1]) * q_next   # (B,unroll)

        q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1) # (B,unroll)
        loss = F.smooth_l1_loss(q_sel, y)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.step_count += 1
        if (self.step_count % self.cfg.target_update)==0:
            self.tgt.load_state_dict(self.net.state_dict())

        return float(loss.item())

# ============================================================
# 🔁 러너: 한 에피소드 실행(데모용) — 네 env에 맞춰 호출
# ============================================================
def run_episode_with_pf_gru(env,
                            agent_self: HybridAgent,
                            agent_other: HybridAgent,
                            pf_self: ParticleFilter,
                            who: str = "chaser",
                            max_steps:int=1000):
    """한 에피소드를 평가 모드로 실행하고, 시퀀스를 리플레이용 포맷으로 반환."""
    assert who in ("chaser","evader")
    pf_self.reset()
    # 리플레이 시퀀스 누적
    xs, acts, rews, dones = [], [], [], []

    # RNN 은닉상태
    h = None
    # 초기 관찰
    ob_s = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
    obs_vec = obs_to_onehot(ob_s)
    belief_feat = pf_self.summarize()

    for t in range(max_steps):
        # 행동 선택
        a_s, h = agent_self.act(obs_vec, belief_feat, h)

        # 상대는 여기선 greedy(데모). 실제 학습에선 상대도 자기 정책/탐험으로.
        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=True)
        obs_vec_o = obs_to_onehot(ob_o)
        belief_o = pf_self.summarize()  # 데모에선 같은 PF를 재사용하거나 별도 PF를 쓰자
        with torch.no_grad():
            q_o, _ = agent_other.net(torch.from_numpy(np.concatenate([obs_vec_o, belief_o],0)).float().view(1,1,-1))
            a_o = int(torch.argmax(q_o, dim=-1).item())

        # 환경 스텝
        if who=="chaser":
            _, (r_c, r_e), done = env.step(a_s, a_o, eval_mode=False)
            r = r_c
        else:
            _, (r_c, r_e), done = env.step(a_o, a_s, eval_mode=False)
            r = r_e

        # 파티클 예측/갱신
        # - 내 움직임 델타(상대좌표에서 빼줄 값) 계산
        my_face_bin = ob_s[3]
        my_move = my_move_delta_from_action(a_s, my_face_bin)
        pf_self.predict(my_move=my_move)
        ob_s_next = env.observe(for_chaser=(who=="chaser"), eval_mode=True)
        pf_self.weight_update(ob_s_next)

        # 누적(다음 스텝 학습용)
        x = np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)
        xs.append(x); acts.append(a_s); rews.append(r); dones.append(1.0 if done else 0.0)

        ob_s = ob_s_next
        obs_vec = obs_to_onehot(ob_s)
        belief_feat = pf_self.summarize()

        if done: break

    # numpy로 정리
    traj = {
        "x": np.stack(xs,0).astype(np.float32),
        "act": np.array(acts, dtype=np.int64),
        "rew": np.array(rews, dtype=np.float32),
        "done": np.array(dones, dtype=np.float32)
    }
    return traj

# ============================================================
# 📈 전이 그래프 생성 + "사본" 프루닝 + 저장
#    (학습에는 절대 사용하지 말 것!)
# ============================================================
def build_transition_graph_from_bins(bin_traj: List[int]) -> nx.DiGraph:
    G = nx.DiGraph()
    # 노드 등록
    for b in bin_traj:
        if b not in G: G.add_node(b)
    # 엣지 카운트
    for i in range(len(bin_traj)-1):
        u,v = bin_traj[i], bin_traj[i+1]
        if G.has_edge(u,v):
            G[u][v]["count"] += 1
        else:
            G.add_edge(u,v, count=1)
    return G

def save_graph_pruned_copy_png(G: nx.DiGraph, cfg: HybridConfig, path_png:str, title:str):
    draw_graph_pruned_copy(
        G, path_png,
        min_edge_count=cfg.viz_min_edge_count,
        topk_per_node=cfg.viz_topk_per_node,
        k_core=cfg.viz_k_core,
        keep_largest=cfg.viz_keep_largest,
        title=title
    )

# ============================================================
# 🧷 데모 메인 (네 프로젝트에 맞게 교체해서 사용)
# ============================================================
if __name__ == "__main__":
    # ---- 여기는 설명용 데모 스텁입니다. ----
    # 실제로는 네 프로젝트의 env/ACTIONS를 import 해서 사용하세요.
    # from your_project import TwoAgentTagEnv, Config, BeliefIndexer, ACTIONS
    print("[hybrid] This is a plug-in module. Import and integrate into your training loop.")
