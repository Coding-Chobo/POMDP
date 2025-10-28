# -*- coding: utf-8 -*-
# Hybrid PF+DRQN (CUDA check + PF=200 with ESS + opponent eps-greedy=0.05)
# - Step guard per episode: 1,000
# - No heavy graphs (speed focus). Save meta/log JSON only.
# - Alternating phases (chaser ↔ evader)
# - Outputs: outputs/applied/*.pt + .meta.json, outputs/logs/*.json

import os, time, json, math, random, threading, shutil
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------ Paths ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
APPLIED_DIR = os.path.join(OUTPUT_DIR, "applied")
SNAP_DIR    = os.path.join(OUTPUT_DIR, "snapshots")
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(APPLIED_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ----------------------- Configs -----------------------
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
    action_cost_rot: float = 0.0003
    action_cost_look: float = 0.00035
    action_cost_focus: float = 0.00045
    action_cost_strafe: float = 0.00015
    action_cost_back: float = 0.00015
    action_cost_spin: float = 0.0003
    action_cost_take_corner: float = 0.0005
    action_cost_collect: float = 0.0003
    action_cost_decoy: float = 0.0008
    # coins
    coin_count: int = 5
    coin_collect_ticks: int = 5
    r_collect_tick: float = 0.2
    r_collect_done: float = 1.0
    r_all_coins_bonus: float = 5.0
    r_coin_detect: float = 0.05
    k_coin_approach: float = 0.002
    r_coin_visible: float = 0.01
    r_duplicate_penalty: float = -0.05
    # threat
    threat_radius: int = 3
    k_threat_avoid: float = 0.003
    # decoy
    decoy_duration: int = 3
    r_decoy_success: float = 0.5
    decoy_slip_add: float = 0.08
    decoy_noise_mult: float = 2.0

@dataclass
class HybridConfig:
    # PF
    pf_num_particles: int = 200              # ↓ 계산량
    pf_process_noise: float = 0.75
    pf_likelihood_ang_sigma: float = 1.2
    pf_likelihood_dist_sigma: float = 1.2
    pf_topk_modes: int = 3                   # (x,y) 모드 3개
    # enc inputs
    # one-hot: ang16 + dist10 + far1 + face8 + coin4 + see2 = 41
    obs_dim_onehot: int = 41
    # belief summary: mean(2)+mean_dist(1)+var_tr(1)+entropy(1)+topk(2*3)=6 → 총 11
    belief_feat_dim: int = 11
    # DRQN
    hidden: int = 128
    n_actions: int = 15
    gamma: float = 0.997
    lr: float = 3e-4
    n_step: int = 1
    # BPTT
    burn_in: int = 16
    unroll: int = 32
    target_update: int = 2000
    # Replay/Batch
    replay_capacity: int = 50_000
    batch_size: int = 64                     # ↑ GPU 활용
    seq_len: int = 48                        # burn_in+unroll (안전 마진)
    # ε
    eps_start: float = 0.20
    eps_final: float = 0.02
    eps_decay: float = 0.9995
    # opponent exploration
    opponent_eps: float = 0.05               # 요청사항

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ----------------------- Helpers -----------------------
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
    ("COLLECT", ()),
    ("DECOY", ()),
]
NUM_ACTIONS = len(ACTIONS)

# ----------------------- Environment -----------------------
import math
class TwoAgentTagEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg; self.n = cfg.grid_size
        self.look_bonus = {"chaser":0, "evader":0}
        self.coins=set(); self.coin_ticks={}
        self.evader_collected=0; self.coins_seen=set()
        self.decoy_timer=0

    def reset(self):
        while True:
            c=(random.randrange(self.n), random.randrange(self.n))
            e=(random.randrange(self.n), random.randrange(self.n))
            if manhattan(c,e) >= self.n//2: break
        self.state=(c,0,e,4)
        self.look_bonus={"chaser":0,"evader":0}
        self.decoy_timer=0; self.evader_collected=0; self.coins_seen.clear()
        self.coins.clear(); self.coin_ticks.clear()
        forbid={self.state[0], self.state[2]}
        while len(self.coins)<self.cfg.coin_count:
            p=(random.randrange(self.n), random.randrange(self.n))
            if p not in forbid and p not in self.coins:
                self.coins.add(p); self.coin_ticks[p]=0
        return self.state

    def _delta_face_to(self, face_idx, dx, dy):
        desired = angle_bin_from_dxdy16(dx, dy) % 8
        diff = (desired - face_idx) % 8
        if diff == 0: return 0
        return +1 if diff <= 4 else -1

    def _apply_action(self, pos, face_idx, who: str, a_idx: int, other_pos):
        kind, param = ACTIONS[a_idx]
        n=self.n; new_pos, new_face=pos, face_idx
        if kind=="WAIT": pass
        elif kind.startswith("ROT"): new_face=(new_face+param[0])%8
        elif kind=="SPIN": new_face=(new_face+4)%8
        elif kind=="FORWARD":
            fdx,fdy=dir_to_vec(new_face); new_pos=add_pos(new_pos,(fdx,fdy),n)
        elif kind=="STRAFE_L":
            sdx,sdy=dir_to_vec(new_face-2); new_pos=add_pos(new_pos,(sdx,sdy),n)
        elif kind=="STRAFE_R":
            sdx,sdy=dir_to_vec(new_face+2); new_pos=add_pos(new_pos,(sdx,sdy),n)
        elif kind=="BACKSTEP":
            bdx,bdy=dir_to_vec(new_face+4); new_pos=add_pos(new_pos,(bdx,bdy),n)
        elif kind=="LOOK":
            key="chaser" if who=="chaser" else "evader"
            self.look_bonus[key]=max(self.look_bonus[key], 1)
        elif kind=="FOCUS":
            key="chaser" if who=="chaser" else "evader"
            turns=max(2, self.look_bonus[key]); self.look_bonus[key]=turns
        elif kind=="TAKE_CORNER":
            dx,dy = other_pos[0]-pos[0], other_pos[1]-pos[1]
            step=self._delta_face_to(new_face,dx,dy); new_face=(new_face+step)%8
            fdx,fdy=dir_to_vec(new_face); new_pos=add_pos(new_pos,(fdx,fdy),n)
        elif kind in ("COLLECT","DECOY"):
            pass
        return new_pos, new_face

    def _nearest_coin_dist(self, epos):
        if not self.coins: return None, 99
        dists=[(p, manhattan(epos,p)) for p in self.coins]
        p,d=min(dists, key=lambda x:x[1]); return p,d

    def step(self, a_c:int, a_e:int, eval_mode=False):
        (c,fc,e,fe)=self.state
        prev_d = manhattan(c,e)

        if self.look_bonus["chaser"]>0: self.look_bonus["chaser"]-=1
        if self.look_bonus["evader"]>0: self.look_bonus["evader"]-=1
        if self.decoy_timer>0: self.decoy_timer-=1

        slip_c = self.cfg.slip + (self.cfg.decoy_slip_add if self.decoy_timer>0 else 0.0)
        slip_e = self.cfg.slip
        if not eval_mode:
            if random.random()<slip_c: a_c=random.randrange(NUM_ACTIONS)
            if random.random()<slip_e: a_e=random.randrange(NUM_ACTIONS)

        c_next,fc_next = self._apply_action(c,fc,"chaser",a_c,e)
        e_next,fe_next = self._apply_action(e,fe,"evader",a_e,c)

        # decoy
        r_decoy=0.0
        if ACTIONS[a_e][0]=="DECOY":
            if manhattan(e_next,c_next) <= self.cfg.fov_radius:
                self.decoy_timer=self.cfg.decoy_duration
                r_decoy=self.cfg.r_decoy_success

        # coin
        r_collect_tick=0.0; r_collect_done=0.0; dup_penalty=0.0
        if ACTIONS[a_e][0]=="COLLECT":
            if e_next in self.coins:
                self.coin_ticks[e_next]+=1
                r_collect_tick=self.cfg.r_collect_tick
                if self.coin_ticks[e_next]>=self.cfg.coin_collect_ticks:
                    self.coins.remove(e_next); del self.coin_ticks[e_next]
                    self.evader_collected+=1
                    r_collect_done=self.cfg.r_collect_done
            else:
                dup_penalty=self.cfg.r_duplicate_penalty

        self.state=(c_next,fc_next,e_next,fe_next)
        done_capture = (manhattan(c_next,e_next) <= self.cfg.capture_radius)
        done_allcoins = (self.evader_collected >= self.cfg.coin_count)
        done = done_capture or done_allcoins

        # base rewards
        r_c = 1.0 if done_capture else 0.0
        r_e = (-1.0 if done_capture else 0.01)

        # coin shaping
        nearest_p, nearest_d_now = self._nearest_coin_dist(e_next)
        if nearest_p is not None and manhattan(e_next,nearest_p) <= self.cfg.fov_radius:
            if nearest_p not in self.coins_seen:
                r_e += self.cfg.r_coin_detect; self.coins_seen.add(nearest_p)
            r_e += self.cfg.r_coin_visible
        _, nearest_d_prev = self._nearest_coin_dist(e)
        if nearest_d_prev != 99 and nearest_d_now != 99:
            r_e += self.cfg.k_coin_approach * (nearest_d_prev - nearest_d_now)
        r_e += r_collect_tick + r_collect_done + r_decoy + dup_penalty
        if done_allcoins: r_e += self.cfg.r_all_coins_bonus

        # generic shaping
        r_c -= self.cfg.time_penalty; r_e -= self.cfg.time_penalty
        new_d = manhattan(c_next,e_next); delta_d = prev_d - new_d
        r_c += self.cfg.k_distance*(delta_d)
        r_e += self.cfg.k_distance*(-delta_d)
        far = new_d > self.cfg.fov_radius
        if not far: r_c += self.cfg.k_los
        else:       r_e += self.cfg.k_los
        kind_c,_=ACTIONS[a_c]; kind_e,_=ACTIONS[a_e]
        if kind_c=="LOOK":  r_c += self.cfg.k_info
        if kind_c=="FOCUS": r_c += 0.75*self.cfg.k_info
        if kind_e=="LOOK":  r_e += self.cfg.k_info
        if kind_e=="FOCUS": r_e += 0.75*self.cfg.k_info

        if new_d <= self.cfg.threat_radius:
            if kind_e not in ("COLLECT","DECOY"): r_e += self.cfg.k_threat_avoid
            elif kind_e=="COLLECT":               r_e -= self.cfg.k_threat_avoid

        def act_cost(kind:str)->float:
            if   kind.startswith("ROT"): return self.cfg.action_cost_rot
            elif kind=="LOOK":           return self.cfg.action_cost_look
            elif kind=="FOCUS":          return self.cfg.action_cost_focus
            elif kind.startswith("STRAFE"): return self.cfg.action_cost_strafe
            elif kind=="BACKSTEP":       return self.cfg.action_cost_back
            elif kind=="SPIN":           return self.cfg.action_cost_spin
            elif kind=="TAKE_CORNER":    return self.cfg.action_cost_take_corner
            elif kind=="COLLECT":        return self.cfg.action_cost_collect
            elif kind=="DECOY":          return self.cfg.action_cost_decoy
            return 0.0
        r_c -= act_cost(kind_c); r_e -= act_cost(kind_e)
        if done_capture: r_c = 1.0; r_e = -1.0

        return self.state, (r_c, r_e), done

    # observation (discrete bins)
    def _dist_bin(self, d, cuts=(1,2,4,6,9,13,18,24)):
        for i,c in enumerate(cuts):
            if d <= c: return i
        return len(cuts)

    def _coin_dist_bin(self, d):
        if d<=2: return 0
        if d<=5: return 1
        if d<=9: return 2
        return 3

    def observe(self, for_chaser: bool, eval_mode=False) -> Tuple[int,int,bool,int,int,int]:
        (c,fc,e,fe) = self.state
        me_pos, me_face = (c,fc) if for_chaser else (e,fe)
        ot_pos = e if for_chaser else c
        key = "chaser" if for_chaser else "evader"

        bonus_turns = self.look_bonus[key]
        extra_noise_mult = (self.cfg.decoy_noise_mult if (for_chaser and self.decoy_timer>0) else 1.0)

        if not eval_mode and bonus_turns>=2:
            eff_fov=int(self.cfg.fov_radius*1.25); base_noise=self.cfg.obs_noise*0.5
        elif not eval_mode and bonus_turns>=1:
            eff_fov=int(self.cfg.fov_radius*1.5); base_noise=self.cfg.obs_noise*0.5
        else:
            eff_fov=self.cfg.fov_radius; base_noise=self.cfg.obs_noise
        eff_noise = 0.0 if eval_mode else base_noise*extra_noise_mult

        dx,dy = ot_pos[0]-me_pos[0], ot_pos[1]-me_pos[1]
        d=abs(dx)+abs(dy); far = d>eff_fov
        if far:
            ang_bin=0; dist_bin=9
        else:
            ang_bin=angle_bin_from_dxdy16(dx,dy)
            dist_bin=self._dist_bin(d)
            if random.random()<eff_noise: ang_bin=(ang_bin+random.choice([-1,1,2]))%16
            if random.random()<eff_noise: dist_bin=max(0, min(8, dist_bin+random.choice([-1,1])))
        my_face_bin = me_face

        nearest_p, nearest_d = self._nearest_coin_dist(e)
        coin_dist_bin=3; see_coin_flag=0
        if nearest_p is not None:
            coin_dist_bin=self._coin_dist_bin(nearest_d)
            if nearest_d <= eff_fov:
                see_coin_flag=1
                if not eval_mode and random.random() < eff_noise: see_coin_flag = 1 - see_coin_flag
        return (ang_bin, dist_bin, far, my_face_bin, coin_dist_bin, see_coin_flag)

# ----------------------- Obs One-hot -----------------------
def obs_to_onehot(obs: Tuple[int,int,bool,int,int,int]) -> np.ndarray:
    ang, dist, far, face, coin_bin, see = obs
    def oh(i,n):
        a=np.zeros(n, dtype=np.float32); a[int(i)%n]=1.0; return a
    v = [
        oh(ang,16),
        oh(9 if far else dist,10),
        np.array([1.0 if far else 0.0], dtype=np.float32),
        oh(face,8),
        oh(coin_bin,4),
        oh(see,2)
    ]
    return np.concatenate(v, axis=0)  # 41

# ----------------------- Particle Filter -----------------------
class ParticleFilter:
    def __init__(self, cfg: HybridConfig, grid_size:int, fov_radius:int, seed:int=0):
        self.cfg=cfg; self.n=grid_size; self.fov=fov_radius
        self.rng=np.random.default_rng(seed)
        self.num=cfg.pf_num_particles
        self.p=None; self.w=None
        self.reset()

    def reset(self):
        low=-self.n+1; high=self.n-1
        self.p = self.rng.integers(low, high+1, size=(self.num,2)).astype(np.float32)
        self.w = np.ones(self.num, dtype=np.float32)/self.num

    def predict(self, my_move:Tuple[int,int]=(0,0)):
        dx,dy = my_move
        self.p[:,0] -= dx; self.p[:,1] -= dy
        jitter = self.rng.normal(0.0, self.cfg.pf_process_noise, size=self.p.shape).astype(np.float32)
        self.p += jitter
        self.p[:,0] = np.clip(self.p[:,0], -self.n+1, self.n-1)
        self.p[:,1] = np.clip(self.p[:,1], -self.n+1, self.n-1)

    def _angle_bin16(self, dx, dy):
        if dx==0 and dy==0: return 0
        a=math.atan2(dy,dx); 
        if a<0: a+=2*math.pi
        return int((a + (2*math.pi/16)/2)//(2*math.pi/16)) % 16

    def _dist_bin10(self, d):
        cuts=[1,2,4,6,9,13,18,24]
        for i,c in enumerate(cuts):
            if d<=c: return i
        return len(cuts)

    def weight_update(self, obs: Tuple[int,int,bool,int,int,int]):
        ang, dist, far, _, _, _ = obs
        dx = self.p[:,0]; dy=self.p[:,1]
        dist_mh = np.abs(dx)+np.abs(dy)
        pred_far = dist_mh > self.fov
        pred_ang = np.array([self._angle_bin16(dx[i],dy[i]) if not pred_far[i] else 0
                             for i in range(self.num)], dtype=np.int32)
        pred_dist = np.array([9 if pred_far[i] else self._dist_bin10(dist_mh[i])
                              for i in range(self.num)], dtype=np.int32)

        def ang_circ_delta(a,b):
            d=abs((a-b)%16); return min(d, 16-d)

        ang_err  = np.array([ang_circ_delta(pred_ang[i], ang) for i in range(self.num)], dtype=np.float32)
        dist_err = np.abs(pred_dist - (9 if far else dist)).astype(np.float32)

        sa=self.cfg.pf_likelihood_ang_sigma
        sd=self.cfg.pf_likelihood_dist_sigma
        like = np.exp(-(ang_err**2)/(2*(sa**2))) * np.exp(-(dist_err**2)/(2*(sd**2))) + 1e-8

        self.w *= like
        s=float(self.w.sum(dtype=np.float64))
        if (not np.isfinite(s)) or (s<=0.0):
            self.w.fill(1.0/self.num)
        else:
            self.w = (self.w / s).astype(np.float32)

        # --- ESS resample (요청사항) ---
        ess = 1.0 / float((self.w.astype(np.float64)**2).sum())
        if ess < 0.5 * self.num:
            self._systematic_resample()

    def _systematic_resample(self):
        N=self.num
        positions=(np.arange(N, dtype=np.float64) + float(self.rng.random())) / float(N)
        cumsum=np.cumsum(self.w, dtype=np.float64)
        if cumsum[-1] <= 0.0 or (not np.isfinite(cumsum[-1])):
            self.w.fill(1.0/N); cumsum=np.cumsum(self.w, dtype=np.float64)
        cumsum[-1]=1.0
        idx=np.searchsorted(cumsum, positions, side='left')
        idx=np.minimum(idx, N-1)
        self.p=self.p[idx]; self.w.fill(1.0/N)

    def summarize(self, topk:int=None)->np.ndarray:
        if topk is None: topk=self.cfg.pf_topk_modes
        mean=(self.w[:,None]*self.p).sum(axis=0)
        var_trace=(self.w[:,None]*((self.p-mean)**2)).sum(axis=0).sum()
        mean_dist=(np.abs(self.p).sum(axis=1)*self.w).sum()
        w=np.clip(self.w, 1e-12, 1.0)
        entropy=float(-(w*np.log(w)).sum()/math.log(len(w)))
        idx=np.argsort(-self.w)[:topk]; top=self.p[idx]
        s=float(self.n)
        feats=[mean[0]/s, mean[1]/s, mean_dist/(2*s), min(1.0, var_trace/(s*s)), entropy]
        for i in range(topk):
            if i < len(top): feats.extend([float(top[i,0]/s), float(top[i,1]/s)])
            else: feats.extend([0.0, 0.0])
        return np.array(feats, dtype=np.float32)  # len=11

# ----------------------- DRQN (dueling) -----------------------
class DRQN(nn.Module):
    def __init__(self, in_dim:int, hidden:int, n_actions:int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU())
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.val = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Linear(128,1))
        self.adv = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Linear(128,n_actions))
    def forward(self, x_seq:torch.Tensor, h0:Optional[torch.Tensor]=None):
        z=self.enc(x_seq)
        out,hT=self.gru(z, h0)
        V=self.val(out); A=self.adv(out)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, hT

# ----------------------- Replay (uniform, fast) -----------------------
class SeqReplay:
    def __init__(self, capacity:int, seq_len:int):
        self.capacity=capacity; self.seq_len=seq_len
        self.data=[]; self.ptr=0
    def push_episode(self, traj:Dict[str,np.ndarray]):
        if len(self.data)<self.capacity: self.data.append(traj)
        else:
            self.data[self.ptr]=traj; self.ptr=(self.ptr+1)%self.capacity
    def sample_batch(self, batch:int, burn_in:int, unroll:int):
        if len(self.data)<batch: return None
        out=[]
        seqs=random.sample(self.data, batch)
        need=burn_in+unroll+1
        for ep in seqs:
            T=len(ep["act"])
            if T<need: return None
            start=random.randint(0, T-need)
            sl=slice(start, start+need)
            out.append({k:v[sl] for k,v in ep.items()})
        return out

# ----------------------- Agent -----------------------
class HybridAgent:
    def __init__(self, hy:HybridConfig, device="cpu"):
        self.cfg=hy
        in_dim = hy.obs_dim_onehot + hy.belief_feat_dim
        self.net = DRQN(in_dim, hy.hidden, hy.n_actions).to(device)
        self.tgt = DRQN(in_dim, hy.hidden, hy.n_actions).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=hy.lr)
        self.device=device
        self.eps=hy.eps_start
        self.step_count=0
        # CUDA warmup (GPU가 실제 사용중인지 초기 한번 돌려봄)
        if device.startswith("cuda"):
            with torch.no_grad():
                dummy = torch.randn(2, 5, in_dim, device=device)
                _ , _ = self.net(dummy)
                torch.cuda.synchronize()

    def decay_eps(self):
        self.eps = max(self.cfg.eps_final, self.eps*self.cfg.eps_decay)

    def q_eval(self, x:np.ndarray, h):
        xt=torch.from_numpy(x).float().to(self.device).view(1,1,-1)
        with torch.no_grad():
            q,h2=self.net(xt, h)
        return q,h2

    def act_eps(self, x:np.ndarray, h):
        q,h2=self.q_eval(x,h)
        if random.random() < self.eps: a=random.randrange(q.shape[-1])
        else: a=int(torch.argmax(q, dim=-1).item())
        return a,h2

    def act_greedy(self, x:np.ndarray, h):
        q,h2=self.q_eval(x,h)
        a=int(torch.argmax(q, dim=-1).item()); return a,h2

    def act_with_eps(self, x:np.ndarray, h, eps_override:float=None):
        q,h2=self.q_eval(x,h)
        eps = self.eps if eps_override is None else eps_override
        if random.random() < eps: a=random.randrange(q.shape[-1])
        else: a=int(torch.argmax(q, dim=-1).item())
        return a,h2

    def train_batch(self, batch, gamma:float):
        if batch is None: return 0.0
        device=self.device
        burn,unroll = self.cfg.burn_in, self.cfg.unroll

        def to_t(name):
            X=np.stack([b[name] for b in batch],0)
            return torch.from_numpy(X).float().to(device)

        x=to_t("x")
        a=torch.from_numpy(np.stack([b["act"] for b in batch],0)).long().to(device)
        r=torch.from_numpy(np.stack([b["rew"] for b in batch],0)).float().to(device)
        d=torch.from_numpy(np.stack([b["done"] for b in batch],0)).float().to(device)

        with torch.no_grad():
            _,h = self.net(x[:, :burn, :])

        q_on,_ = self.net(x[:, burn:-1, :], h)  # (B,unroll,A)
        q_sel = q_on.gather(-1, a[:, burn:-1].unsqueeze(-1)).squeeze(-1)  # (B,unroll)

        with torch.no_grad():
            q_on_next,_  = self.net(x[:, burn+1:, :], h)
            a_star = torch.argmax(q_on_next, dim=-1)
            q_tgt_next,_ = self.tgt(x[:, burn+1:, :], h)
            q_next = q_tgt_next.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)

            target = r[:, burn:burn+unroll] + gamma * (1.0 - d[:, burn:burn+unroll]) * q_next

        loss_el = F.smooth_l1_loss(q_sel, target, reduction='mean')

        self.optim.zero_grad(); loss_el.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.step_count += 1
        if (self.step_count % self.cfg.target_update)==0:
            self.tgt.load_state_dict(self.net.state_dict())

        return float(loss_el.item())

# ----------------------- Utils -----------------------
def my_move_delta_from_action(action_idx:int, my_face_bin:int)->Tuple[int,int]:
    if action_idx==6:   vx,vy = dir_to_vec(my_face_bin)          # FORWARD
    elif action_idx==7: vx,vy = dir_to_vec((my_face_bin-2)%8)    # STRAFE_L
    elif action_idx==8: vx,vy = dir_to_vec((my_face_bin+2)%8)    # STRAFE_R
    elif action_idx==9: vx,vy = dir_to_vec((my_face_bin+4)%8)    # BACKSTEP
    else:               vx,vy = (0,0)
    return (vx,vy)

class StopFlag:
    def __init__(self): self.flag=False
def stop_listener(sf: StopFlag):
    try:
        s=input("학습 중지하려면 'q' 입력 후 ENTER: ").strip().lower()
        if s=='q': sf.flag=True
    except: pass

def save_model(applied_path_pt:str, meta_path_json:str, agent:HybridAgent, role:str):
    torch.save(agent.net.state_dict(), applied_path_pt)
    meta={"role":role, "arch":"DRQN(Dueling)-GRU",
          "obs_dim":agent.cfg.obs_dim_onehot, "belief_dim":agent.cfg.belief_feat_dim,
          "hidden":agent.cfg.hidden, "n_actions":agent.cfg.n_actions,
          "gamma":agent.cfg.gamma, "eps":agent.eps, "step_count":agent.step_count,
          "file":os.path.basename(applied_path_pt)}
    with open(meta_path_json,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)

# ----------------------- Episode run & learn -----------------------
def play_episode_and_learn(env:TwoAgentTagEnv, who:str,
                           agent_self:HybridAgent, agent_other:HybridAgent,
                           pf_self:ParticleFilter, pf_other:ParticleFilter,
                           replay:SeqReplay, max_steps:int=1000,
                           opponent_eps:float=0.05):
    assert who in ("chaser","evader")
    env.reset(); pf_self.reset(); pf_other.reset()

    xs=[]; acts=[]; rews=[]; dones=[]
    h_self=None; h_oth=None

    ob_s=env.observe(for_chaser=(who=="chaser"), eval_mode=False)
    ob_o=env.observe(for_chaser=(who!="chaser"), eval_mode=False)
    sum_r_c=0.0; sum_r_e=0.0; captured=False

    for _ in range(max_steps):
        x_self = np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0)
        x_oth  = np.concatenate([obs_to_onehot(ob_o), pf_other.summarize()], axis=0)

        a_s, h_self = agent_self.act_eps(x_self, h_self)               # 학습 대상
        a_o, h_oth  = agent_other.act_with_eps(x_oth, h_oth, opponent_eps)  # ← ε=0.05 (요청)

        if who=="chaser":
            _, (r_c,r_e), done = env.step(a_s, a_o, eval_mode=False)
            r_who = r_c
        else:
            _, (r_c,r_e), done = env.step(a_o, a_s, eval_mode=False)
            r_who = r_e

        sum_r_c += r_c; sum_r_e += r_e

        # PF update (self)
        my_face_bin = ob_s[3]
        pf_self.predict(my_move=my_move_delta_from_action(a_s, my_face_bin))
        ob_s_next = env.observe(for_chaser=(who=="chaser"), eval_mode=False)
        pf_self.weight_update(ob_s_next)

        xs.append(np.concatenate([obs_to_onehot(ob_s), pf_self.summarize()], axis=0))
        acts.append(a_s); rews.append(r_who); dones.append(1.0 if done else 0.0)

        # PF update (opponent)
        my_face_bin_o = ob_o[3]
        pf_other.predict(my_move=my_move_delta_from_action(a_o, my_face_bin_o))
        ob_o = env.observe(for_chaser=(who!="chaser"), eval_mode=False)
        pf_other.weight_update(ob_o)

        ob_s = ob_s_next

        if done:
            captured=True; break

    traj={"x":np.stack(xs,0).astype(np.float32),
          "act":np.array(acts,dtype=np.int64),
          "rew":np.array(rews,dtype=np.float32),
          "done":np.array(dones,dtype=np.float32)}
    replay.push_episode(traj)

    batch = replay.sample_batch(agent_self.cfg.batch_size, agent_self.cfg.burn_in, agent_self.cfg.unroll)
    loss = agent_self.train_batch(batch, agent_self.cfg.gamma) if batch else 0.0
    agent_self.decay_eps()
    return len(xs), sum_r_c, sum_r_e, captured, float(loss)

# ----------------------- Train loop -----------------------
def train_forever(
    env_cfg:EnvConfig = EnvConfig(),
    hycfg:HybridConfig = HybridConfig(),
    SHOW_EVERY:int = 1000,         # episodes per phase (1k 마다 JSON 저장)
    PRINT_EVERY:int = 100,
    MAX_STEPS:int = 1000           # per episode
):
    # ---------- CUDA availability & info ----------
    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    if cuda_ok:
        print("CUDA version:", torch.version.cuda, "Device:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"
    # ---------------------------------------------

    env = TwoAgentTagEnv(env_cfg)
    chaser = HybridAgent(hycfg, device=device)
    evader = HybridAgent(hycfg, device=device)
    pf_ch = ParticleFilter(hycfg, env_cfg.grid_size, env_cfg.fov_radius, seed=0)
    pf_ev = ParticleFilter(hycfg, env_cfg.grid_size, env_cfg.fov_radius, seed=1)
    replay = SeqReplay(hycfg.replay_capacity, hycfg.seq_len)

    sf=StopFlag()
    threading.Thread(target=stop_listener, args=(sf,), daemon=True).start()

    outer=1
    while not sf.flag:
        # -------- Phase A: Chaser updates --------
        steps_acc=r_c_acc=r_e_acc=0.0; cap_cnt=0; avg_loss=0.0; loss_n=0
        best_score=-1e18; best_ep=None

        t0=time.time()
        for ep in range(1, SHOW_EVERY+1):
            steps,sr_c,sr_e,cap,loss = play_episode_and_learn(
                env,"chaser", chaser, evader, pf_ch, pf_ev, replay,
                max_steps=MAX_STEPS, opponent_eps=hycfg.opponent_eps)
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            avg_loss += loss; loss_n += 1
            if sr_c > best_score: best_score=sr_c; best_ep={"sum_r":sr_c,"steps":steps}
            if (ep % PRINT_EVERY)==0:
                print(f"[Chaser] outer {outer} {ep}/{SHOW_EVERY} eps={chaser.eps:.3f} loss={(avg_loss/max(1,loss_n)):.4f}")
            if sf.flag: break
        dur=time.time()-t0

        # save applied + snapshot + log
        applied_pt = os.path.join(APPLIED_DIR, "chaser_policy.pt")
        applied_meta = os.path.join(APPLIED_DIR, "chaser_policy.meta.json")
        save_model(applied_pt, applied_meta, chaser, "chaser")
        stamp=time.strftime("%Y%m%d-%H%M%S")
        torch.save(chaser.net.state_dict(), os.path.join(SNAP_DIR, f"chaser_outer{outer}_{stamp}.pt"))
        log={
            "phase":"chaser","outer":outer,"episodes":SHOW_EVERY,"timestamp":stamp,
            "avg_steps":steps_acc/SHOW_EVERY,
            "avg_reward_chaser":r_c_acc/SHOW_EVERY,
            "avg_reward_evader":r_e_acc/SHOW_EVERY,
            "capture_rate":cap_cnt/SHOW_EVERY,
            "epsilon_chaser":chaser.eps,"epsilon_evader":evader.eps,
            "avg_loss":(avg_loss/max(1,loss_n)),
            "duration_sec":dur,
            "best_episode":best_ep
        }
        with open(os.path.join(LOGS_DIR, f"log_chaser_outer{outer}_{stamp}.json"),"w",encoding="utf-8") as f:
            json.dump(log,f,ensure_ascii=False,indent=2)

        if sf.flag: break

        # -------- Phase B: Evader updates --------
        steps_acc=r_c_acc=r_e_acc=0.0; cap_cnt=0; avg_loss=0.0; loss_n=0
        best_score=-1e18; best_ep=None
        t0=time.time()
        for ep in range(1, SHOW_EVERY+1):
            steps,sr_c,sr_e,cap,loss = play_episode_and_learn(
                env,"evader", evader, chaser, pf_ev, pf_ch, replay,
                max_steps=MAX_STEPS, opponent_eps=hycfg.opponent_eps)
            steps_acc += steps; r_c_acc += sr_c; r_e_acc += sr_e; cap_cnt += int(cap)
            avg_loss += loss; loss_n += 1
            if sr_e > best_score: best_score=sr_e; best_ep={"sum_r":sr_e,"steps":steps}
            if (ep % PRINT_EVERY)==0:
                print(f"[Evader] outer {outer} {ep}/{SHOW_EVERY} eps={evader.eps:.3f} loss={(avg_loss/max(1,loss_n)):.4f}")
            if sf.flag: break
        dur=time.time()-t0

        applied_pt = os.path.join(APPLIED_DIR, "evader_policy.pt")
        applied_meta = os.path.join(APPLIED_DIR, "evader_policy.meta.json")
        save_model(applied_pt, applied_meta, evader, "evader")
        stamp=time.strftime("%Y%m%d-%H%M%S")
        torch.save(evader.net.state_dict(), os.path.join(SNAP_DIR, f"evader_outer{outer}_{stamp}.pt"))
        log={
            "phase":"evader","outer":outer,"episodes":SHOW_EVERY,"timestamp":stamp,
            "avg_steps":steps_acc/SHOW_EVERY,
            "avg_reward_chaser":r_c_acc/SHOW_EVERY,
            "avg_reward_evader":r_e_acc/SHOW_EVERY,
            "capture_rate":cap_cnt/SHOW_EVERY,
            "epsilon_chaser":chaser.eps,"epsilon_evader":evader.eps,
            "avg_loss":(avg_loss/max(1,loss_n)),
            "duration_sec":dur,
            "best_episode":best_ep
        }
        with open(os.path.join(LOGS_DIR, f"log_evader_outer{outer}_{stamp}.json"),"w",encoding="utf-8") as f:
            json.dump(log,f,ensure_ascii=False,indent=2)

        outer += 1

# ----------------------- Main -----------------------
if __name__ == "__main__":
    # 가벼운 기본값(속도 우선). 필요하면 SHOW_EVERY만 올려서 사용.
    train_forever(
        env_cfg=EnvConfig(),
        hycfg=HybridConfig(),
        SHOW_EVERY=10000,      # 1,000 에피소드마다 JSON/스냅샷 저장
        PRINT_EVERY=10,      # 진행 로그
        MAX_STEPS=1000        # 에피소드 스텝가드
    )
