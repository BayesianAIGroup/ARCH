
from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from matplotlib.colors import LogNorm
from typing import Optional, Tuple
from datetime import datetime
from dateutil import parser
from utils import _parse_ts
from utils import get_my_data
from utils import LossConfig
from utils import *
from enum import Enum, auto
from dataclasses import dataclass
from torchdiffeq import odeint
#from EventsFM2 import *
from flow_backbone import *
import pickle
#from generate_hawkes_delayed import *
from generate_self_correcting import *

def wrap_up_data(seqs):
    res = []
    for seq in seqs:
        instance = {}
        instance['length'] = len(seq)
        seq = np.concatenate([np.array([0]), seq])
        instance['deltas'] = torch.tensor(np.diff(seq), dtype=torch.float32)
        instance['locs'] = torch.zeros(instance['length'], 2, dtype=torch.float32)
        res.append(instance)
    return res

def predict_error(
    model_time,
    model_loc,
    dl,
    num_steps_time: int = 100,
    num_steps_loc: int = 100,
    use_fixed_noise: bool = True,
):
    """Integrate the temporal flow from 0 to 1, then the spatial flow conditioned on predicted deltas. Return time RMSE/relative RMSE/relative L2 in real delta units and mean spatial distance in the target location space."""
    device = next(model_time.parameters()).device
    model_time.eval()
    model_loc.eval()

    delta_pred_list = []
    delta_gt_list = []
    loc_pred_list = []
    loc_gt_list = []

    with torch.no_grad():
        for batch in dl:
            # move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            deltas_tgt = batch["deltas_tgt"]      # [B, T]
            locs_tgt   = batch["locs_tgt"]        # [B, T, loc_dim]
            seq_mask   = batch["seq_mask"]        # [B, T]
            predict_mask = batch["predict_mask"]  # [B, T]

            B, T = deltas_tgt.shape
            loc_dim = model_time.loc_dim

            if not predict_mask.any():
                continue

            # --- padding masks ---
            key_padding_mask = ~seq_mask  # [B, T]

            if "deltas_mem" in batch and "locs_mem" in batch and "mem_seq_mask" in batch:
                deltas_mem = batch["deltas_mem"]           # [B, T_mem]
                locs_mem   = batch["locs_mem"]             # [B, T_mem, loc_dim]
                mem_seq_mask = batch["mem_seq_mask"]       # [B, T_mem]
            else:
                deltas_mem = torch.cat([torch.zeros(B, 1, device=device), deltas_tgt[:, :-1]], dim=1)
                locs_mem   = torch.cat([torch.zeros(B, 1, loc_dim, device=device), locs_tgt[:, :-1, :]], dim=1)
                mem_seq_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), seq_mask[:, :-1]], dim=1)

            mem_key_padding = ~mem_seq_mask

            update_mask = predict_mask & seq_mask  # [B, T]

            # -----------------------
            # (1) time model sampling
            # -----------------------
            z_time = deltas_tgt.unsqueeze(-1).clone()  # [B, T, 1]
            if use_fixed_noise and ("noise_time" in batch):
                noise_time = batch["noise_time"].unsqueeze(-1)  # [B, T, 1]
            else:
                noise_time = torch.randn_like(z_time)

            z_time[update_mask] = noise_time[update_mask]

            locs_s_dummy = torch.zeros(B, T, loc_dim, device=device)

            s_val = 0.0
            ds = 1.0 / num_steps_time
            for _ in range(num_steps_time):
                s = torch.full((B, T), s_val, device=device)
                deltas_s = z_time[..., 0]  # [B, T]

                v_time, _ = model_time(
                    deltas_noisy=deltas_s,
                    locs_noisy=locs_s_dummy,
                    flow_t=s,
                    deltas=deltas_tgt,
                    locs=locs_tgt,
                    key_padding_mask=key_padding_mask,
                    deltas_mem=deltas_mem,
                    locs_mem=locs_mem,
                    memory_key_padding_mask=mem_key_padding,
                    condition_mask=None,
                    auto_regressive=True,
                    predict_loc_only=False,
                )

                z_time[update_mask] = z_time[update_mask] + ds * v_time[update_mask].unsqueeze(-1)
                s_val += ds

            delta_pred_gen = z_time[..., 0]  # [B, T]

            # time RMSE (real space)
            mask = update_mask.clone()
            if mask.shape[1] > 0:
                mask[:, 0] = False

            delta_gt_   = deltas_tgt[mask]
            delta_pred_ = delta_pred_gen[mask]

            if model_time.do_log_transform:
                delta_gt_real   = model_time.log_transform_inverse(delta_gt_)   / model_time.rescale
                delta_pred_real = model_time.log_transform_inverse(delta_pred_) / model_time.rescale
            else:
                delta_gt_real   = delta_gt_   / model_time.rescale
                delta_pred_real = delta_pred_ / model_time.rescale

            if model_time.delta_min is not None and model_time.delta_max is not None:
                delta_min = model_time.delta_min.to(device)
                delta_max = model_time.delta_max.to(device)
                scale = (delta_max - delta_min)
                delta_gt_real   = delta_gt_real   * scale + delta_min
                delta_pred_real = delta_pred_real * scale + delta_min

            delta_pred_list.append(delta_pred_real.detach().cpu())
            delta_gt_list.append(delta_gt_real.detach().cpu())

            # -----------------------
            # (2) loc model sampling
            # -----------------------
            z_loc = locs_tgt.clone()  # [B, T, loc_dim]
            if use_fixed_noise and ("noise_loc" in batch):
                noise_loc = batch["noise_loc"]  # [B, T, loc_dim]
            else:
                noise_loc = torch.randn_like(z_loc)

            z_loc[update_mask] = noise_loc[update_mask]

            s_val = 0.0
            ds_loc = 1.0 / num_steps_loc
            for _ in range(num_steps_loc):
                s = torch.full((B, T), s_val, device=device)

                deltas_for_loc = delta_pred_gen  # [B, T]

                _, v_loc = model_loc(
                    deltas_noisy=deltas_for_loc,
                    locs_noisy=z_loc,
                    flow_t=s,
                    deltas=deltas_for_loc,
                    locs=locs_tgt,
                    key_padding_mask=key_padding_mask,
                    deltas_mem=deltas_mem,
                    locs_mem=locs_mem,
                    memory_key_padding_mask=mem_key_padding,
                    condition_mask=None,
                    auto_regressive=True,
                    predict_loc_only=True,
                )

                z_loc[update_mask] = z_loc[update_mask] + ds_loc * v_loc[update_mask]
                s_val += ds_loc

            loc_pred = z_loc[mask]     # [N, loc_dim]
            loc_gt   = locs_tgt[mask]  # [N, loc_dim]
            loc_pred_list.append(loc_pred.detach().cpu())
            loc_gt_list.append(loc_gt.detach().cpu())

    if len(delta_pred_list) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    delta_pred_all = torch.cat(delta_pred_list, dim=0)
    delta_gt_all   = torch.cat(delta_gt_list,   dim=0)
    time_rmse = torch.sqrt(torch.mean((delta_pred_all - delta_gt_all) ** 2))
    rel_rmse = time_rmse / (delta_gt_all.std() + 1e-10)
    rel_l2 = torch.sqrt(torch.sum((delta_pred_all - delta_gt_all) ** 2)) / (torch.sqrt(torch.sum(delta_gt_all ** 2)) + 1e-10)

    loc_pred_all = torch.cat(loc_pred_list, dim=0)  # [N, loc_dim]
    loc_gt_all   = torch.cat(loc_gt_list,   dim=0)
    loc_euclid = torch.sqrt(((loc_pred_all - loc_gt_all) ** 2).sum(dim=-1)).mean()

    return time_rmse.item(), rel_rmse.item(), rel_l2.item(), loc_euclid.item()


def simualte_data(num_seqs = 5000):
    res = []
    for _ in range(num_seqs):
        gaussian = np.random.normal()
        exponential = np.random.exponential(1)
        instance = {}
        instance['length'] = 1
        instance['deltas'] = torch.tensor([exponential], dtype=torch.float32)
        instance['noise'] = torch.tensor([gaussian], dtype=torch.float32)
        instance['locs'] = torch.zeros(instance['length'], 2, dtype=torch.float32)
        res.append(instance)
    return res


#conditioned on 1st n-1 events, get intensity function for the nth event
def get_intensity_function(model_time: SpatioTemporalTraceformer, instance, n: int, device='cuda'):
    L = instance['length']
    if n >= L or n < 0:
        raise ValueError("n must be in [1, length-1]")
    Nstep = 100
    target_delta = instance['deltas'][n].item()
    #x1 = torch.linspace(-2, 4, steps=Nstep).to(device)  # time deltas to query, shape [Nstep]
    x1 = torch.linspace(0, target_delta , steps=Nstep).to(device)  # time deltas to query, shape [Nstep]
    cif_input = x1.clone()
    if model_time.do_log_transform:
        x1 = model_time.log_transform(torch.tensor(x1, dtype=torch.float32)).to(device)
    deltas_tgt = instance['deltas'][:n+1].unsqueeze(0).repeat(Nstep, 1).to(device)  # [Nstep, n+1]
    if model_time.do_log_transform:
        deltas_tgt = model_time.log_transform(deltas_tgt)
    locs_tgt = instance['locs'][:n+1].unsqueeze(0).repeat(Nstep, 1, 1).to(device)  # [Nstep, n+1, loc_dim]
    deltas_tgt = torch.concat([deltas_tgt, torch.zeros(Nstep, 1, device=device)], dim=-1)  # [Nstep, n+2]
    locs_tgt = torch.concat([locs_tgt, torch.zeros(Nstep, 1, locs_tgt.shape[-1], device=device)], dim=-2)  # [Nstep, n+2, loc_dim]

    seq_mask = torch.ones(Nstep, n+2, dtype=torch.bool, device=device)  # [Nstep, n+2]
    seq_mask[:, -1] = False  # the last one is padding
    mem_seq_mask = torch.ones(Nstep, n+2, dtype=torch.bool, device=device)  # [Nstep, n+2]
    mem_seq_mask[:, 0] = False  # the first one is padding
    mask = torch.zeros(Nstep, n+2, dtype=torch.bool, device=device)  # [Nstep, n+2]
    mask[:, n] = True

    model_time.eval()
    num_steps_time = 100
    B, T = deltas_tgt.shape
    loc_dim = model_time.loc_dim
    key_padding_mask = ~seq_mask
    mem_key_padding = ~mem_seq_mask
    deltas_mem = torch.concat([torch.zeros(B, 1, device=device), deltas_tgt[:, :n+1]], dim=-1)  # [Nstep, n+2]
    locs_mem = torch.concat([torch.zeros(B, 1, loc_dim, device=device), locs_tgt[:, :n+1, :]], dim=-2)  # [Nstep, n+2, loc_dim]

    # ---------- reverse integration ----------
    ll = torch.zeros_like(x1)
    s_val  = 1.0
    ds_time = -1.0 / num_steps_time
    locs_s_dummy  = torch.zeros(B, T, loc_dim, device=device)    # [B, T, loc_dim]
    for _ in range(num_steps_time):
        s = torch.full((B, T), s_val, device=device)  # [B, T]
        x1 = x1.detach().requires_grad_(True)
        deltas_s = torch.concat([deltas_tgt[:, :n], x1.unsqueeze(-1), deltas_tgt[:, n+1:]], dim=-1)# [B, T]
        v_time, _ = model_time(
            deltas_noisy = deltas_s,
            locs_noisy = locs_s_dummy,
            flow_t = s,
            deltas=deltas_tgt,
            locs=locs_tgt,
            key_padding_mask=key_padding_mask,
            deltas_mem=deltas_mem,
            locs_mem=locs_mem,
            memory_key_padding_mask=mem_key_padding,
            condition_mask=None,
            auto_regressive=True,
            predict_loc_only=False,
        )
        v = v_time[mask]
        div = torch.autograd.grad(
            outputs=v,
            inputs=x1,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]  # [B, T]
        x1 = x1 + ds_time * v.detach()
        ll = ll + ds_time * div.detach()
        s_val += ds_time
    x0 = x1
    log_p0 = -0.5*np.log(2 * np.pi) - 0.5 * x0.pow(2)
    log_prob = log_p0 + ll
    if model_time.do_log_transform:
        log_prob = log_prob - (cif_input+1e-6).log()
    #cdf = log_prob.exp().cumsum(dim=0) * (cif_input[1]-cif_input[0])
    #log_survival = torch.log(1.0 - cdf + 1e-10)
    log_survival = torch.special.log_ndtr(-x0)
    cif = (log_prob - log_survival).exp()
    return log_prob.detach().cpu().numpy(), cif.detach().cpu().numpy(), cif_input.detach().cpu().numpy(), instance['deltas'][n].item()

def get_intensity_function_dopri5(model_time, instance, n: int, device="cuda", num_points: int = 20,
                                query_deltas=None):
    L = instance["length"]
    if n >= L or n < 0:
        raise ValueError("n must be in [0, length-1]")

    if num_points < 2:
        raise ValueError("num_points must include zero and at least one positive query")
    Nstep = int(num_points)
    target_delta = float(instance["deltas"][n].item())

    # query grid in real-delta space (for plotting / Jacobian correction)
    if query_deltas is None:
        x_real = torch.linspace(0.0, target_delta, steps=Nstep, device=device)
    else:
        # Optional external evaluation grid: zero is retained for the existing
        # ODE bookkeeping and removed from the returned arrays below.
        x_real = torch.as_tensor(query_deltas, dtype=instance["deltas"].dtype, device=device)
        if (x_real.ndim != 1 or x_real.numel() < 2 or not torch.isfinite(x_real).all()
                or x_real[0] != 0 or torch.any(x_real[1:] <= x_real[:-1])
                or x_real[-1] > target_delta):
            raise ValueError("query_deltas must start at zero, strictly increase, and stay inside the interval")
        Nstep = x_real.numel()
    cif_input = x_real.clone()

    # x in model space (maybe log-transform)
    x_init = x_real
    if model_time.do_log_transform:
        x_init = model_time.log_transform(x_init)

    # build tgt (condition) sequences
    deltas_tgt = instance["deltas"][: n + 1].unsqueeze(0).repeat(Nstep, 1).to(device)  # [B, n+1]
    if model_time.do_log_transform:
        deltas_tgt = model_time.log_transform(deltas_tgt)

    locs_tgt = instance["locs"][: n + 1].unsqueeze(0).repeat(Nstep, 1, 1).to(device)  # [B, n+1, loc_dim]

    # append one trailing padding token (like your code)
    deltas_tgt = torch.cat([deltas_tgt, torch.zeros(Nstep, 1, device=device)], dim=-1)           # [B, n+2]
    locs_tgt   = torch.cat([locs_tgt,   torch.zeros(Nstep, 1, locs_tgt.shape[-1], device=device)], dim=1)  # [B, n+2, loc_dim]

    B, T = deltas_tgt.shape
    loc_dim = model_time.loc_dim

    # masks
    seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    seq_mask[:, -1] = False  # last is padding
    key_padding_mask = ~seq_mask

    mem_seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mem_seq_mask[:, 0] = False  # first is padding (dummy)
    mem_key_padding = ~mem_seq_mask

    # which position is the queried token in decoder sequence?
    # you used mask[:, n] = True (query sits at index n)
    query_pos = n
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, query_pos] = True

    # build memory (your original construction)
    deltas_mem = torch.cat([torch.zeros(B, 1, device=device), deltas_tgt[:, : n + 1]], dim=-1)              # [B, n+2]
    locs_mem   = torch.cat([torch.zeros(B, 1, loc_dim, device=device), locs_tgt[:, : n + 1, :]], dim=1)    # [B, n+2, loc_dim]

    locs_s_dummy = torch.zeros(B, T, loc_dim, device=device)

    model_time.eval()

    # ---------------------------
    # Dopri5 reverse integration: s from 1 -> 0
    # state = (x, ll), where x is [B] and ll is [B]
    # ---------------------------
    class ODEFunc(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, s, state):
            # s: scalar tensor
            x, ll = state                      # x: [B], ll: [B]
            # divergence needs grad wrt x
            x = x.detach().requires_grad_(True)

            # build deltas_s by inserting x at query_pos
            # deltas_s: [B, T]
            deltas_s = deltas_tgt.clone()
            deltas_s[:, query_pos] = x

            s_full = torch.full((B, T), float(s.item()), device=device)

            v_time, _ = model_time(
                deltas_noisy=deltas_s,
                locs_noisy=locs_s_dummy,
                flow_t=s_full,
                deltas=deltas_tgt,
                locs=locs_tgt,
                key_padding_mask=key_padding_mask,
                deltas_mem=deltas_mem,
                locs_mem=locs_mem,
                memory_key_padding_mask=mem_key_padding,
                condition_mask=None,
                auto_regressive=True,
                predict_loc_only=False,
            )

            v = v_time[mask]

            div = torch.autograd.grad(
                outputs=v.sum(),
                inputs=x,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]  # [B]

            # dx/ds = v, dll/ds = div
            return v.detach(), div.detach()

    func = ODEFunc()

    # initial at s=1
    x1 = x_init
    ll0 = torch.zeros_like(x1)

    # integrate from 1 -> 0 (reverse)
    t_span = torch.tensor([1.0, 0.0], device=device)

    x_traj, ll_traj = odeint(
        func,
        (x1, ll0),
        t_span,
        method="dopri5",
        rtol=1e-4,
        atol=1e-5,
        options={"max_num_steps": 500},
    )

    x0 = x_traj[-1]     # [B] at s=0
    ll = ll_traj[-1]    # [B]

    # base density log p0(x0) for standard normal
    log_p0 = -0.5 * np.log(2.0 * np.pi) - 0.5 * x0.pow(2)
    log_prob = log_p0 + ll

    # Jacobian correction if you used log-transform on delta
    if model_time.do_log_transform:
        log_prob = log_prob - (cif_input + 1e-6).log()

    # survival in standard normal (same as your code)
    log_survival = torch.special.log_ndtr(-x0)

    cif = (log_prob - log_survival).exp()

    log_prob = log_prob[1:]
    cif = cif[1:]
    cif_input = cif_input[1:]

    return (
        log_prob.detach().cpu().numpy(),
        cif.detach().cpu().numpy(),
        cif_input.detach().cpu().numpy(),
        float(instance["deltas"][n].item()),
    )


def get_logprob_loc_dopri5(model_loc, instance, n: int, delta_query: float, xy_query: torch.Tensor, device="cuda"):
    """Compute log p(x,y | delta_query, history up to n-1) by reverse integration from 1 to 0 with a standard normal base density. xy_query has shape [B, 2]; grid queries can be batched."""
    model_loc.eval()
    xy_query = xy_query.to(device)
    B = xy_query.shape[0]
    loc_dim = 2

    L = instance["length"]
    if n >= L or n < 0:
        raise ValueError("n must be in [0, length-1]")

    # -------------------------
    # -------------------------
    T = n + 2
    query_pos = n

    # deltas in REAL space
    deltas_real = instance["deltas"][: n + 1].unsqueeze(0).repeat(B, 1).to(device)  # [B, n+1]
    deltas_real[:, query_pos] = float(delta_query)

    deltas_tgt = deltas_real
    if getattr(model_loc, "do_log_transform", False):
        deltas_tgt = model_loc.log_transform(deltas_tgt)

    locs_hist = instance["locs"][: n + 1].unsqueeze(0).repeat(B, 1, 1).to(device)  # [B, n+1,2]
    locs_tgt = locs_hist.clone()
    locs_tgt[:, query_pos, :] = 0.0

    # append trailing padding token (like your time intensity code)
    deltas_tgt = torch.cat([deltas_tgt, torch.zeros(B, 1, device=device)], dim=-1)                    # [B, n+2]
    locs_tgt   = torch.cat([locs_tgt,   torch.zeros(B, 1, loc_dim, device=device)], dim=1)            # [B, n+2,2]

    # masks
    seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    seq_mask[:, -1] = False
    key_padding_mask = ~seq_mask

    mem_seq_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mem_seq_mask[:, 0] = False
    mem_key_padding = ~mem_seq_mask

    deltas_mem = torch.cat([torch.zeros(B, 1, device=device), deltas_tgt[:, : n + 1]], dim=-1)              # [B, n+2]
    locs_mem   = torch.cat([torch.zeros(B, 1, loc_dim, device=device), locs_tgt[:, : n + 1, :]], dim=1)     # [B, n+2,2]

    # -------------------------
    # Dopri5 reverse ODE: state = (x, ll)
    # x: [B,2], ll: [B]
    # -------------------------
    class ODEFunc(torch.nn.Module):
        def forward(self, s, state):
            x, ll = state  # x:[B,2]
            x = x.detach().requires_grad_(True)

            locs_noisy = locs_tgt.clone()
            locs_noisy[:, query_pos, :] = x

            s_full = torch.full((B, T), float(s.item()), device=device)

            # loc-only：predict_loc_only=True
            _, v_loc = model_loc(
                deltas_noisy=deltas_tgt,
                locs_noisy=locs_noisy,
                flow_t=s_full,
                deltas=deltas_tgt,
                locs=locs_tgt,
                key_padding_mask=key_padding_mask,
                deltas_mem=deltas_mem,
                locs_mem=locs_mem,
                memory_key_padding_mask=mem_key_padding,
                condition_mask=None,
                auto_regressive=True,
                predict_loc_only=True,
            )

            v = v_loc[:, query_pos, :]  # [B,2]

            g0 = torch.autograd.grad(v[:, 0].sum(), x, retain_graph=True, create_graph=False)[0][:, 0]
            g1 = torch.autograd.grad(v[:, 1].sum(), x, retain_graph=False, create_graph=False)[0][:, 1]
            div = g0 + g1  # [B]

            return v.detach(), div.detach()

    func = ODEFunc()

    x1 = xy_query
    ll0 = torch.zeros(B, device=device)

    t_span = torch.tensor([1.0, 0.0], device=device)
    x_traj, ll_traj = odeint(
        func,
        (x1, ll0),
        t_span,
        method="dopri5",
        rtol=1e-4,
        atol=1e-5,
        options={"max_num_steps": 500},
    )

    x0 = x_traj[-1]      # [B,2] at s=0
    ll = ll_traj[-1]     # [B]

    # base log p0(x0) for N(0,I)
    log_p0 = -0.5 * loc_dim * np.log(2.0 * np.pi) - 0.5 * (x0.pow(2).sum(dim=-1))
    log_prob = log_p0 + ll
    return log_prob  # [B]


def get_pxy_heatmap(model_loc, instance, n: int, delta_query: float, grid_size=40, device="cuda",
                    xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = torch.tensor(
        np.stack([XX.reshape(-1), YY.reshape(-1)], axis=-1),
        dtype=torch.float32,
        device=device,
    )  # [G^2,2]

    logp = get_logprob_loc_dopri5(model_loc, instance, n=n, delta_query=delta_query, xy_query=pts, device=device)
    p = torch.exp(logp).reshape(grid_size, grid_size).detach().cpu().numpy()

    dx = (xlim[1] - xlim[0]) / (grid_size - 1)
    dy = (ylim[1] - ylim[0]) / (grid_size - 1)
    Z = (p.sum() * dx * dy) + 1e-12
    p = p / Z

    return XX, YY, p

def plot_txy_compare(
    model_time,
    model_loc,
    instance,
    n,
    g0_cov=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    g2_cov=np.array([[0.85, 0.0], [0.0, 0.85]], dtype=np.float64),
    alpha=0.2,
    beta=0.2,
    mu=1.0,
    gamma=2.0,
    lamb_max=10.0,
    x_num=51,
    y_num=51,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
    grid_size=40,
    pick_k_times=3,
    device="cuda",
    out_prefix="txy_compare",
):
    """
    For fixed interval n:
      - learned λ_t(Δt) from model_time
      - choose several Δt points
      - learned λ(t,x,y)=λ_t(Δt)*p_learn(x,y|Δt)
      - gt λ(t,x,y) from STSCP
    """
    logp, cif, cif_input, target_delta = get_intensity_function_dopri5(model_time, instance, n=n, device=device)
    lam_t_grid = np.asarray(cif)
    delta_grid = np.asarray(cif_input)

    # build global-time events
    L = int(instance["length"])
    deltas_np = instance["deltas"].detach().cpu().numpy()
    locs_np   = instance["locs"].detach().cpu().numpy()
    t_events = np.cumsum(deltas_np[:L]).astype(np.float64)
    gt_events = np.concatenate([t_events[:, None], locs_np[:L].astype(np.float64)], axis=1)

    t_start = float(np.sum(deltas_np[:n])) if n > 0 else 0.0

    idx_pool = np.arange(len(delta_grid))
    idx_pool = idx_pool[idx_pool > 1] if len(idx_pool) > 2 else idx_pool
    pick_k_times = min(pick_k_times, len(idx_pool))
    choose = np.random.choice(idx_pool, size=pick_k_times, replace=False)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, pick_k_times, figsize=(5 * pick_k_times, 8), dpi=180)
    if pick_k_times == 1:
        axes = np.array(axes).reshape(2, 1)

    eps = 1e-12
    for j, idx in enumerate(choose):
        delta_q = float(delta_grid[idx])
        lam_t = float(lam_t_grid[idx])
        t_query = t_start + delta_q

        # learned
        XX, YY, pxy = get_pxy_heatmap(
            model_loc, instance, n=n, delta_query=delta_q,
            grid_size=grid_size, device=device, xlim=x_lim, ylim=y_lim
        )
        lam_xy_learn = lam_t * pxy

        # GT from STSCP
        XXg, YYg, lam_xy_gt = gt_lambda_txy_heatmap_stscp(
            t_query=t_query,
            events=gt_events,
            alpha=alpha,
            beta=beta,
            mu=mu,
            gamma=gamma,
            g0_cov=g0_cov,
            g2_cov=g2_cov,
            grid_size=grid_size,
            x_num=x_num,
            y_num=y_num,
            x_lim=x_lim,
            y_lim=y_lim,
            lamb_max=lamb_max,
        )

        # shared color norm based on learned
        pos_gt = lam_xy_gt[lam_xy_gt > 0]
        if pos_gt.size == 0:
            vmin, vmax = eps, 1.0
        else:
            vmin = max(pos_gt.min(), eps)
            vmax = max(pos_gt.max(), vmin * 10)

        norm = LogNorm(vmin=vmin, vmax=vmax)

        ax0 = axes[0, j]
        im0 = ax0.imshow(
            lam_xy_learn,
            origin="lower",
            extent=[XX.min(), XX.max(), YY.min(), YY.max()],
            aspect="auto",
            norm=norm,
        )
        ax0.set_title(f"Learned λ(t,x,y)\nΔt={delta_q:.4f}, t={t_query:.4f}")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[1, j]
        im1 = ax1.imshow(
            lam_xy_gt,
            origin="lower",
            extent=[XXg.min(), XXg.max(), YYg.min(), YYg.max()],
            aspect="auto",
            norm=norm,
        )
        ax1.set_title("GT λ(t,x,y) (STSCP)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        hist = gt_events[gt_events[:, 0] < t_query]
        if hist.shape[0] > 0:
            ax0.scatter(hist[:, 1], hist[:, 2], s=8, alpha=0.35)
            ax1.scatter(hist[:, 1], hist[:, 2], s=8, alpha=0.35)

    fig.suptitle(f"STSCP spatio-temporal intensity comparison at interval n={n}", y=0.98)
    fig.tight_layout()
    out = f"{out_prefix}_n{n}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_txy_compare_3d(
    model_time,
    model_loc,
    instance,
    n,
    g0_cov=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    g2_cov=np.array([[0.85, 0.0], [0.0, 0.85]], dtype=np.float64),
    alpha=0.2,
    beta=0.2,
    mu=1.0,
    gamma=2.0,
    lamb_max=10.0,
    x_num=51,
    y_num=51,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
    grid_size=40,
    pick_k_times=4,
    t_quantile_range=(0.15, 0.98),
    include_event_bracket=False,
    sample_event_times=True,
    event_t_range=(50.0, 90.0),
    surface_cmap="YlGnBu",
    surface_alpha=0.88,
    elev=28,
    azim=-60,
    device="cuda",
    out_prefix="txy_compare_3d",
):
    """
    3D surface comparison for fixed interval n:
      - top row: learned lambda(t,x,y)
      - bottom row: GT lambda(t,x,y)
      - can sample columns directly from event times in a target global-time range
    """
    _, cif, cif_input, _ = get_intensity_function_dopri5(model_time, instance, n=n, device=device)
    lam_t_grid = np.asarray(cif)
    delta_grid = np.asarray(cif_input)

    L = int(instance["length"])
    deltas_np = instance["deltas"].detach().cpu().numpy()
    locs_np = instance["locs"].detach().cpu().numpy()
    t_events = np.cumsum(deltas_np[:L]).astype(np.float64)
    gt_events = np.concatenate([t_events[:, None], locs_np[:L].astype(np.float64)], axis=1)

    pick_k_times = max(2, int(pick_k_times))
    query_specs = []

    if sample_event_times:
        t_lo, t_hi = float(event_t_range[0]), float(event_t_range[1])
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
        cand = np.where((t_events >= t_lo) & (t_events <= t_hi))[0]
        if cand.size == 0:
            cand = np.arange(len(t_events))
        k = min(pick_k_times, cand.size)
        choose_event = np.random.choice(cand, size=k, replace=False)
        choose_event = np.sort(choose_event)

        for ev_idx in choose_event:
            n_use = int(ev_idx)
            delta_target = float(deltas_np[n_use])
            _, cif_ev, cif_input_ev, _ = get_intensity_function_dopri5(model_time, instance, n=n_use, device=device)
            lam_grid_ev = np.asarray(cif_ev)
            delta_grid_ev = np.asarray(cif_input_ev)
            idx_use = int(np.argmin(np.abs(delta_grid_ev - delta_target)))
            query_specs.append(("event", n_use, idx_use, float(delta_grid_ev[idx_use]), float(lam_grid_ev[idx_use])))
    else:
        idx_pool = np.arange(len(delta_grid))
        idx_pool = idx_pool[idx_pool > 1] if len(idx_pool) > 2 else idx_pool
        if len(idx_pool) == 0:
            print("No valid time grid for 3D plotting.")
            return

        n_total = int(instance["length"])
        use_bracket = bool(include_event_bracket and (n + 1 < n_total))
        regular_k = max(1, pick_k_times - 2) if use_bracket else pick_k_times
        regular_k = min(regular_k, len(idx_pool))

        q_lo, q_hi = t_quantile_range
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, q_lo, 1.0))
        frac = np.linspace(q_lo, q_hi, regular_k)
        choose = idx_pool[np.round(frac * (len(idx_pool) - 1)).astype(int)]
        choose = np.unique(choose)
        if len(choose) < regular_k:
            for idx in idx_pool:
                if idx not in choose:
                    choose = np.append(choose, idx)
                if len(choose) >= regular_k:
                    break
        choose = np.sort(choose[:regular_k])
        for idx in choose:
            query_specs.append(("regular", n, int(idx), float(delta_grid[idx]), float(lam_t_grid[idx])))

        if use_bracket:
            pre_idx = int(idx_pool[-1]) if len(idx_pool) == 1 else int(idx_pool[-2])
            query_specs.append(("pre-event", n, pre_idx, float(delta_grid[pre_idx]), float(lam_t_grid[pre_idx])))

            _, cif_next, cif_input_next, _ = get_intensity_function_dopri5(model_time, instance, n=n + 1, device=device)
            lam_t_grid_next = np.asarray(cif_next)
            delta_grid_next = np.asarray(cif_input_next)
            idx_pool_next = np.arange(len(delta_grid_next))
            idx_pool_next = idx_pool_next[idx_pool_next > 1] if len(idx_pool_next) > 2 else idx_pool_next
            if len(idx_pool_next) == 0:
                post_idx = int(np.argmin(delta_grid_next))
            else:
                post_idx = int(idx_pool_next[0])
            query_specs.append(("post-event", n + 1, post_idx, float(delta_grid_next[post_idx]), float(lam_t_grid_next[post_idx])))

    import matplotlib.pyplot as plt

    num_cols = len(query_specs)
    fig = plt.figure(figsize=(5 * num_cols, 9), dpi=180)

    zmax = 0.0
    panel_cache = []

    for tag, n_use, idx_use, delta_q, lam_t in query_specs:
        t_start_use = float(np.sum(deltas_np[:n_use])) if n_use > 0 else 0.0
        t_query = t_start_use + delta_q

        XX, YY, pxy = get_pxy_heatmap(
            model_loc,
            instance,
            n=n_use,
            delta_query=delta_q,
            grid_size=grid_size,
            device=device,
            xlim=x_lim,
            ylim=y_lim,
        )
        lam_xy_learn = lam_t * pxy

        XXg, YYg, lam_xy_gt = gt_lambda_txy_heatmap_stscp(
            t_query=t_query,
            events=gt_events,
            alpha=alpha,
            beta=beta,
            mu=mu,
            gamma=gamma,
            g0_cov=g0_cov,
            g2_cov=g2_cov,
            grid_size=grid_size,
            x_num=x_num,
            y_num=y_num,
            x_lim=x_lim,
            y_lim=y_lim,
            lamb_max=lamb_max,
        )

        zmax = max(zmax, float(np.max(lam_xy_learn)), float(np.max(lam_xy_gt)))
        panel_cache.append((tag, n_use, delta_q, t_query, XX, YY, lam_xy_learn, XXg, YYg, lam_xy_gt))

    zmax = max(zmax, 1e-6)
    zmin = 0.0

    for j, (tag, n_use, delta_q, t_query, XX, YY, lam_xy_learn, XXg, YYg, lam_xy_gt) in enumerate(panel_cache):
        ax0 = fig.add_subplot(2, num_cols, j + 1, projection="3d")
        ax0.plot_surface(
            XX, YY, lam_xy_learn,
            cmap=surface_cmap,
            vmin=zmin,
            vmax=zmax,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=surface_alpha,
        )
        ax0.set_title(f"Learned [{tag}]\\nn={n_use}, Δt={delta_q:.4f}, t={t_query:.4f}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("lambda")
        ax0.set_zlim(0.0, zmax * 1.05)
        ax0.view_init(elev=elev, azim=azim)

        ax1 = fig.add_subplot(2, num_cols, num_cols + j + 1, projection="3d")
        ax1.plot_surface(
            XXg, YYg, lam_xy_gt,
            cmap=surface_cmap,
            vmin=zmin,
            vmax=zmax,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=surface_alpha,
        )
        ax1.set_title(f"GT (STSCP) [{tag}]")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("lambda")
        ax1.set_zlim(0.0, zmax * 1.05)
        ax1.view_init(elev=elev, azim=azim)

    fig.suptitle(f"3D STSCP intensity comparison at interval n={n}", y=0.98)
    fig.tight_layout()
    out = f"{out_prefix}_n{n}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    print("3D queried global t:", [f"{v[3]:.4f} ({v[0]})" for v in panel_cache])

def _gaussian2d_pdf_points(points, center, cov):
    """
    points: [N,2]
    center: [2]
    cov: [2,2]
    """
    inv_cov = np.linalg.inv(cov)
    sqrt_inv_det = 1.0 / np.sqrt(np.linalg.det(cov))
    delta = points - center[None, :]
    expo = -0.5 * np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    return (1.0 / (2.0 * np.pi)) * sqrt_inv_det * np.exp(expo)


def _build_stscp_grid(x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), x_num=51, y_num=51):
    xs = np.linspace(x_lim[0], x_lim[1], x_num)
    ys = np.linspace(y_lim[0], y_lim[1], y_num)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    s_grids = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=-1)
    return xs, ys, XX, YY, s_grids


def _stoi_from_grid(s, x_num, y_num):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    ij = np.round(s * np.array([[x_num - 1, y_num - 1]])).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, x_num - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, y_num - 1)
    return (ij[:, 0] + ij[:, 1] * x_num).astype(int)


def _precompute_stscp_kernels(
    g0_cov,
    g2_cov,
    mu,
    gamma,
    x_num=51,
    y_num=51,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
):
    xs, ys, XX, YY, s_grids = _build_stscp_grid(x_lim, y_lim, x_num, y_num)
    num_point = x_num * y_num

    # g0 centered at (0.5, 0.5), normalized as in STSCPDataset
    g0_raw = _gaussian2d_pdf_points(s_grids, np.array([0.5, 0.5]), g0_cov)
    g0_mat = g0_raw / np.sum(g0_raw) * x_num * y_num * mu

    g2_mats = np.zeros((num_point, num_point), dtype=np.float64)
    center_ref = np.array([0.5, 0.5], dtype=np.float64)

    for i, s in enumerate(s_grids):
        local_cov = g2_cov / (gamma * np.sum((s - center_ref) ** 2) + 1.0)
        g2_raw = _gaussian2d_pdf_points(s_grids, s, local_cov)
        g2_mats[i] = g2_raw / np.sum(g2_raw) * x_num * y_num

    return {
        "xs": xs,
        "ys": ys,
        "XX": XX,
        "YY": YY,
        "s_grids": s_grids,
        "g0_mat": g0_mat,
        "g2_mats": g2_mats,
        "x_num": x_num,
        "y_num": y_num,
    }


def stscp_lambda_vector(t_query, events, *, alpha, beta, mu, gamma, g0_cov, g2_cov,
                        x_num=51, y_num=51, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), lamb_max=10.0):
    """
    Full-grid STSCP intensity vector λ_St(t|H), matching STSCPDataset.lamb_St
    """
    cache = _precompute_stscp_kernels(
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        mu=mu,
        gamma=gamma,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
    )
    g0_mat = cache["g0_mat"]
    g2_mats = cache["g2_mats"]

    mu_vec = mu * np.ones(x_num * y_num, dtype=np.float64)

    hist = events[events[:, 0] < t_query] if (events is not None and len(events) > 0) else np.zeros((0, 3))

    if hist.shape[0] == 0:
        lam = mu_vec * np.exp(g0_mat * beta * t_query)
        return np.minimum(lamb_max, lam), cache

    idx = _stoi_from_grid(hist[:, 1:3], x_num, y_num)
    suppress = np.sum(g2_mats[idx], axis=0)
    lam = mu_vec * np.exp(g0_mat * beta * t_query - alpha * suppress)
    lam = np.minimum(lamb_max, lam)
    return lam, cache


def gt_lambda_t_given_history_stscp(t_grid, events, *, alpha, beta, mu, gamma, g0_cov, g2_cov,
                                    x_num=51, y_num=51, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), lamb_max=10.0):
    """
    GT λ(t|H) for STSCP, matching lamb_t = mean over all spatial grid intensities.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    lam_t = np.empty_like(t_grid, dtype=np.float64)

    for j, t in enumerate(t_grid):
        lam_vec, _ = stscp_lambda_vector(
            t,
            events,
            alpha=alpha,
            beta=beta,
            mu=mu,
            gamma=gamma,
            g0_cov=g0_cov,
            g2_cov=g2_cov,
            x_num=x_num,
            y_num=y_num,
            x_lim=x_lim,
            y_lim=y_lim,
            lamb_max=lamb_max,
        )
        lam_t[j] = np.mean(lam_vec)

    return lam_t


def gt_lambda_txy_heatmap_stscp(t_query, events, *, alpha, beta, mu, gamma, g0_cov, g2_cov,
                                grid_size=40, x_num=51, y_num=51,
                                x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), lamb_max=10.0):
    """
    GT λ(t,x,y|H) for STSCP.
    Query points are mapped to nearest STSCP grid cells.
    """
    lam_vec, cache = stscp_lambda_vector(
        t_query,
        events,
        alpha=alpha,
        beta=beta,
        mu=mu,
        gamma=gamma,
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
        lamb_max=lamb_max,
    )

    xs = np.linspace(x_lim[0], x_lim[1], grid_size)
    ys = np.linspace(y_lim[0], y_lim[1], grid_size)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=-1)
    idx = _stoi_from_grid(pts, x_num, y_num)
    lam_xy = lam_vec[idx].reshape(grid_size, grid_size)

    return XX, YY, lam_xy


def _pickle_load_cpu_compat(path):
    import io
    import sys
    import pickle as _pickle
    import numpy.core as npcore
    import torch.storage

    sys.modules.setdefault("numpy._core", npcore)
    sys.modules.setdefault("numpy._core.multiarray", npcore.multiarray)
    sys.modules.setdefault("numpy._core.numeric", npcore.numeric)

    old_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    try:
        with open(path, "rb") as f:
            return _pickle.load(f)
    finally:
        torch.storage._load_from_bytes = old_loader


def _deepstpp_processing_baseline(data, scales):
    t_ints = data["intensity"]
    s_sampling_ints = data["S_Intensities"]
    test_idx = data["indices"]

    test_t_lamb = torch.cat(t_ints, dim=0).unsqueeze(-1) / scales[-1]
    test_lamb_s_sampling = torch.cat(s_sampling_ints, dim=0) / np.prod(scales[:2])

    seq_idx_cur = 0
    test_s_sampling = []
    test_int_t = []
    temp_s_sampling = []
    temp_int_t = []

    for i in range(test_idx.shape[0]):
        seq_idx = test_idx[i][0]
        if seq_idx != seq_idx_cur:
            seq_idx_cur = seq_idx
            if len(temp_s_sampling) > 0:
                test_s_sampling.append(torch.cat(temp_s_sampling, dim=0).cpu())
            if len(temp_int_t) > 0:
                test_int_t.append(torch.cat(temp_int_t, dim=0).cpu())
            temp_s_sampling = []
            temp_int_t = []
        temp_s_sampling.append(test_lamb_s_sampling[i].unsqueeze(0))
        temp_int_t.append(test_t_lamb[i])

    if len(temp_s_sampling) > 0:
        test_s_sampling.append(torch.cat(temp_s_sampling, dim=0).cpu())
    if len(temp_int_t) > 0:
        test_int_t.append(torch.cat(temp_int_t, dim=0).cpu())

    return test_s_sampling, test_int_t


def _nstpp_processing_baseline(data):
    t_ints = data["t_ints"]
    st_sampling_ints = data["test_s_ints"]
    t_mask_ints = data["mask"]

    test_st_sampling = []
    test_int_t = []
    for i in range(len(t_ints)):
        for j in range(t_ints[i].shape[0]):
            sp_valid = st_sampling_ints[i][j][t_mask_ints[i][j].to(bool)].permute(0, 2, 1, 3)
            st_seq = torch.cat([sp_valid[0]] + [sp_valid[k][1:] for k in range(1, sp_valid.shape[0])], dim=0).prod(dim=-1)[:-1].cpu()
            t_seq = t_ints[i][j][t_mask_ints[i][j].to(bool)].cpu()
            test_st_sampling.append(st_seq)
            test_int_t.append(t_seq)

    return test_st_sampling, test_int_t


def _instance_from_abs_sequence(seq):
    L = seq.shape[0]
    time_stamps = seq[:, 0]
    locs = seq[:, 1:]
    ts0 = np.concatenate([np.array([0.0]), time_stamps])
    instance = {
        "length": L,
        "deltas": torch.tensor(np.diff(ts0), dtype=torch.float32),
        "locs": torch.tensor(locs, dtype=torch.float32),
        "noise_time": torch.randn(L, dtype=torch.float32),
        "noise_loc": torch.randn(L, 2, dtype=torch.float32),
    }
    return instance


def plot_sc_3d_with_baselines(
    model_time,
    model_loc,
    seq_idx=0,
    pick_k_times=5,
    time_range=(50.0, 90.0),
    first_pick_mode="random",
    baseline_root="Spatial_Temporal_Plotting",
    g0_cov=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    g2_cov=np.array([[0.85, 0.0], [0.0, 0.85]], dtype=np.float64),
    alpha=0.2,
    beta=0.2,
    mu=1.0,
    gamma=2.0,
    lamb_max=10.0,
    x_num=51,
    y_num=51,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
    grid_size=30,
    surface_cmap="YlGnBu",
    surface_alpha=0.9,
    deep_nstpp_adaptive=True,
    arch_adaptive=True,
    gt_adaptive=True,
    autostpp_adaptive=True,
    col_width=3.6,
    row_height=2.95,
    row_label_fontsize=14,
    col_title_fontsize=11,
    show_suptitle=False,
    elev=28,
    azim=-60,
    device="cuda",
    out_file="stscp_3d_gt_arch_baselines.png",
):
    """
    Row order is fixed as:
      1) GT
      2) ARCH (our model)
      3) AutoSTPP
      4) DeepSTPP
      5) NSTPP
    Uses the same sequence and the same picked event times for all rows.
    """
    root = data_path(baseline_root)
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(__file__), root)

    test_path = os.path.join(root, "test_dataset", "stscp_out", "data_test.pkl")
    auto_path = os.path.join(root, "baselines", "stscp", "AUTOSTPP.pkl")
    deep_path = os.path.join(root, "baselines", "stscp", "DEEPSTPP.pkl")
    nst_path = os.path.join(root, "baselines", "stscp", "NSTPP.pkl")

    test_data = _pickle_load_cpu_compat(test_path)
    auto_raw = _pickle_load_cpu_compat(auto_path)
    deep_raw = _pickle_load_cpu_compat(deep_path)
    nst_raw = _pickle_load_cpu_compat(nst_path)

    deep_s, deep_t = _deepstpp_processing_baseline(deep_raw, auto_raw["scales"])
    nst_st, _ = _nstpp_processing_baseline(nst_raw)

    auto_cond = auto_raw["test_st_sampling"][seq_idx] / auto_raw["test_t_ints"][seq_idx].unsqueeze(-1)
    # Match Spatial_Temporal_Plotting2/data_plotting_new.py: DeepSTPP uses the
    # spatial S_Intensities scaled by spatial scales only, without dividing by
    # the temporal intensity again.
    deep_cond = deep_s[seq_idx]
    nst_cond_full = nst_st[seq_idx]  # full length, baseline script uses [10:]

    seq = np.asarray(test_data[seq_idx], dtype=np.float32)
    instance = _instance_from_abs_sequence(seq)
    L = int(instance["length"])

    deltas_np = instance["deltas"].detach().cpu().numpy()
    locs_np = instance["locs"].detach().cpu().numpy()
    t_events = np.cumsum(deltas_np[:L]).astype(np.float64)
    gt_events = np.concatenate([t_events[:, None], locs_np[:L].astype(np.float64)], axis=1)

    # Align with baseline implementation: use event indices >= 10
    time_seq = seq[10:, 0].astype(np.float64)
    idx_pool = np.arange(len(time_seq))
    t_lo, t_hi = float(time_range[0]), float(time_range[1])
    if t_hi < t_lo:
        t_lo, t_hi = t_hi, t_lo
    in_range = idx_pool[(time_seq >= t_lo) & (time_seq <= t_hi)]
    if len(in_range) == 0:
        in_range = idx_pool

    k = min(int(pick_k_times), len(in_range))
    if k <= 0:
        print("No valid time index to plot.")
        return

    if str(first_pick_mode).lower() == "center":
        t_mid = 0.5 * (t_lo + t_hi)
        choose_local = np.array([int(in_range[np.argmin(np.abs(time_seq[in_range] - t_mid))])], dtype=int)
        if k > 1:
            rest_pool = in_range[in_range != choose_local[0]]
            n_rest = min(k - 1, len(rest_pool))
            rest_pick = np.random.choice(rest_pool, size=n_rest, replace=False)
            choose_local = np.concatenate([choose_local, np.sort(rest_pick)])
    else:
        # default: pure random k points (same behavior as previous version)
        choose_local = np.sort(np.random.choice(in_range, size=k, replace=False))

    xs = np.linspace(x_lim[0], x_lim[1], grid_size)
    ys = np.linspace(y_lim[0], y_lim[1], grid_size)
    XX_base, YY_base = np.meshgrid(xs, ys, indexing="xy")

    cols = []
    for local_idx in choose_local:
        n_use = int(local_idx + 10)
        t_query = float(time_seq[local_idx])

        # ARCH at event time for this n
        _, cif, cif_input, _ = get_intensity_function_dopri5(model_time, instance, n=n_use, device=device)
        lam_t_grid = np.asarray(cif)
        delta_grid = np.asarray(cif_input)
        delta_target = float(deltas_np[n_use])
        idx_use = int(np.argmin(np.abs(delta_grid - delta_target)))
        delta_q = float(delta_grid[idx_use])
        lam_t = float(lam_t_grid[idx_use])

        XXa, YYa, pxy = get_pxy_heatmap(
            model_loc, instance, n=n_use, delta_query=delta_q,
            grid_size=grid_size, device=device, xlim=x_lim, ylim=y_lim
        )
        arch_map = lam_t * pxy

        XXg, YYg, gt_map = gt_lambda_txy_heatmap_stscp(
            t_query=t_query,
            events=gt_events,
            alpha=alpha,
            beta=beta,
            mu=mu,
            gamma=gamma,
            g0_cov=g0_cov,
            g2_cov=g2_cov,
            grid_size=grid_size,
            x_num=x_num,
            y_num=y_num,
            x_lim=x_lim,
            y_lim=y_lim,
            lamb_max=lamb_max,
        )

        # Baseline maps at the same local index (aligned by baseline script design)
        # Baseline plotting code used torch.meshgrid(samples, samples), whose
        # default order is ij-style. GT/ARCH use np.meshgrid(..., indexing="xy"),
        # so transpose baseline maps before plotting them on the xy mesh.
        auto_map = auto_cond[local_idx].reshape(grid_size, grid_size).detach().cpu().numpy().T
        deep_map = deep_cond[local_idx].reshape(grid_size, grid_size).detach().cpu().numpy().T
        nst_map = nst_cond_full[local_idx + 10].reshape(grid_size, grid_size).detach().cpu().numpy().T

        cols.append({
            "t_query": t_query,
            "n_use": n_use,
            "GT": (XXg, YYg, gt_map),
            "ARCH": (XXa, YYa, arch_map),
            "AutoSTPP": (XX_base, YY_base, auto_map),
            "DeepSTPP": (XX_base, YY_base, deep_map),
            "NSTPP": (XX_base, YY_base, nst_map),
        })

    import matplotlib.pyplot as plt

    row_names = ["GT", "ARCH", "AutoSTPP", "DeepSTPP", "NSTPP"]
    n_rows = len(row_names)
    n_cols = len(cols)
    fig = plt.figure(figsize=(max(2.8, col_width) * n_cols, row_height * n_rows), dpi=200)

    for c, col in enumerate(cols):
        # Per-column scale so weaker methods are still visible, while rows remain comparable at same t.
        all_max = max(float(np.max(col[r][2])) for r in row_names)
        vmax = max(all_max, 1e-8)
        vmin = 0.0
        for r, name in enumerate(row_names):
            XXp, YYp, Zp = col[name]
            if gt_adaptive and name == "GT":
                row_vmax = max(float(np.max(Zp)), 1e-8)
            elif arch_adaptive and name == "ARCH":
                row_vmax = max(float(np.max(Zp)), 1e-8)
            elif autostpp_adaptive and name == "AutoSTPP":
                row_vmax = max(float(np.max(Zp)), 1e-8)
            elif deep_nstpp_adaptive and name in ("DeepSTPP", "NSTPP"):
                row_vmax = max(float(np.max(Zp)), 1e-8)
            else:
                row_vmax = vmax
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            ax.set_anchor("NW")
            try:
                ax.set_box_aspect((1, 1, 0.58), zoom=1.04)
            except TypeError:
                ax.set_box_aspect((1, 1, 0.58))
            ax.plot_surface(
                XXp, YYp, Zp,
                cmap=surface_cmap,
                vmin=vmin,
                vmax=row_vmax,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=surface_alpha,
            )
            if r == 0:
                ax.set_title(f"t={col['t_query']:.1f}", fontsize=col_title_fontsize, fontweight="bold", pad=2.0)
            ax.set_xlabel("x", labelpad=-2, fontsize=10, fontweight="bold")
            ax.set_ylabel("y", labelpad=-2, fontsize=10, fontweight="bold")
            ax.set_zlabel("λ", labelpad=-1, fontsize=10, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
            ax.tick_params(axis="z", which="major", labelsize=8, pad=0)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                tick_label.set_fontweight("bold")
            zlim_scale = 1.0 if name == "DeepSTPP" else 1.0
            ax.set_zlim(0.0, row_vmax * 1.05 * zlim_scale)
            ax.view_init(elev=elev, azim=azim)

    # Stronger row headers on the left margin (more visible than per-axis zlabels).
    for r, name in enumerate(row_names):
        y = 1.0 - (r + 0.5) / n_rows
        fig.text(
            0.078,
            y,
            name,
            va="center",
            ha="right",
            fontsize=row_label_fontsize,
            fontweight="bold",
        )

    if show_suptitle:
        fig.suptitle(
            f"STSCP 3D Comparison (same seq={seq_idx}, same times in [{t_lo:.1f}, {t_hi:.1f}]): "
            "GT, ARCH, AutoSTPP, DeepSTPP, NSTPP",
            y=0.985,
            fontsize=12,
        )
    fig.subplots_adjust(left=0.09, right=0.998, top=0.94, bottom=0.04, wspace=-0.05, hspace=0.0)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")
    print("Picked global times:", [f"{c['t_query']:.4f}" for c in cols])


def load_data_3d(filename: str):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)

    res = []
    max_len = 0
    for seq in loaded_data:
        time_stamps = seq[:, 0]
        locs = seq[:, 1:]  # [L,2]

        instance = {}
        L = seq.shape[0]
        instance['length'] = L

        ts0 = np.concatenate([np.array([0.0]), time_stamps])
        instance['deltas'] = torch.tensor(np.diff(ts0), dtype=torch.float32)   # [L]
        instance['locs']   = torch.tensor(locs, dtype=torch.float32)           # [L,2]

        instance['noise_time'] = torch.randn(L, dtype=torch.float32)
        instance['noise_loc']  = torch.randn(L, 2, dtype=torch.float32)

        res.append(instance)
        max_len = max(max_len, L)

    return res, max_len


def train(device='cuda', save_best=True, epochs=None):
    train_data, max_len_tr = load_data_3d(data_path('stscp_out/data_train.pkl'))
    test_data,  max_len_te = load_data_3d(data_path('stscp_out/data_test.pkl'))
    max_len = max(max_len_tr, max_len_te) + 1
    print(len(train_data), len(test_data), max_len_tr, max_len_te)

    # ---------------------------
    # 1) build two models
    # ---------------------------
    model_time = SpatioTemporalTraceformer(
        loc_dim=2,
        d_model=32,
        n_heads=1,
        n_layers=4,
        d_ff=32,
        max_len=max_len,
        dropout=0,
    ).to(device)

    model_loc = SpatioTemporalTraceformer(
        loc_dim=2,
        d_model=32,
        n_heads=1,
        n_layers=4,
        d_ff=32,
        max_len=max_len,
        dropout=0,
    ).to(device)

    # make sure both models use the SAME settings
    model_time.set_log_transform(True)
    model_loc.set_log_transform(True)

    model_time.set_RN_train(False)
    model_time.set_AR_train(True)

    model_loc.set_RN_train(False)
    model_loc.set_AR_train(True)

    # ---------------------------
    # 2) optimizers
    # ---------------------------
    lr = 1e-3
    max_epochs = 160 if epochs is None else epochs
    batch_size = 128

    opt_time = torch.optim.Adam(model_time.parameters(), lr=lr)
    opt_loc  = torch.optim.Adam(model_loc.parameters(),  lr=lr)

    loss_cfg = STLossConfig(time_loss_weight=0.5, loc_loss_weight=1.0, rc_loss_weight=1.5)

    # IMPORTANT: use model_time.collate_with_mask so delta log-transform is consistent
    dl_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: model_time.collate_with_mask(b, max_len),
    )
    dl_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: model_time.collate_with_mask(b, max_len),
    )

    # ---------------------------
    # 3) train
    # ---------------------------
    for epoch in range(max_epochs):
        model_time.train()
        model_loc.train()

        total_loss_sum = 0.0
        time_loss_sum  = 0.0
        loc_loss_sum   = 0.0
        n_batch = 0

        for _, batch in enumerate(dl_train, start=1):
            # move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # shared random condition mask for BOTH models (so the "fixed"/"predicted" tokens align)
            rc = (torch.rand_like(batch["deltas_tgt"]) > 0.3)
            rc = rc.to(device)
            # keep rc only on valid tokens; and usually force first token to be conditioned
            if "seq_mask" in batch:
                rc = rc & batch["seq_mask"]
                rc[:, 0] = True
            batch["random_condition"] = rc

            # compute losses from two different models
            loss_time, t_metrics = model_time.compute_time_losses_flow(batch, loss_cfg)
            loss_loc,  l_metrics = model_loc.compute_loc_losses_flow(batch, loss_cfg)

            # backward separately (clean + safe)
            opt_time.zero_grad(set_to_none=True)
            opt_loc.zero_grad(set_to_none=True)

            loss_time.backward()
            loss_loc.backward()

            # optional grad clip (uncomment if needed)
            # torch.nn.utils.clip_grad_norm_(model_time.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(model_loc.parameters(),  max_norm=1.0)

            opt_time.step()
            opt_loc.step()

            total_loss = loss_time.item() + loss_loc.item()
            total_loss_sum += total_loss
            time_loss_sum  += loss_time.item()
            loc_loss_sum   += loss_loc.item()
            n_batch += 1

        avg_total = total_loss_sum / max(1, n_batch)
        avg_time  = time_loss_sum  / max(1, n_batch)
        avg_loc   = loc_loss_sum   / max(1, n_batch)

        # evaluate time model only (your predict_error is time-only right now)
        time_rmse, rel_rmse, rel_l2, loc_euclid = predict_error(
            model_time, model_loc, dl_test,
            num_steps_time=10, num_steps_loc=10
        )

        print(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"Train Total: {avg_total:.4f} | "
            f"Train Time: {avg_time:.4f} | "
            f"Train Loc: {avg_loc:.4f} | "
            f"Test Time RMSE: {time_rmse:.4f} | "
            f"Test Loc Mean Euclid: {loc_euclid:.4f} | "
            f"Test Time Rel RMSE: {rel_rmse:.4f} | "
            f"Test Time Rel L2: {rel_l2:.4f}"

        )

    # ---------------------------
    # 4) save
    # ---------------------------
    if save_best:
        torch.save(model_time.state_dict(), 'best_model_time_SC.pth')
        torch.save(model_loc.state_dict(),  'best_model_loc_SC.pth')
        print("Saved: best_model_time_SC.pth, best_model_loc_SC.pth")


def visualize(device="cuda", model_time_path="best_model_time_SC.pth", model_loc_path="best_model_loc_SC.pth"):
    train_data, max_len_tr = load_data_3d(data_path('stscp_out/data_train.pkl'))
    test_data,  max_len_te = load_data_3d(data_path('stscp_out/data_test.pkl'))
    max_len = max(max_len_tr, max_len_te) + 1

    instance = test_data[20]

    # --- build two models ---
    model_time = SpatioTemporalTraceformer(
        loc_dim=2, d_model=32, n_heads=1, n_layers=4, d_ff=32, max_len=max_len, dropout=0
    ).to(device)
    model_loc = SpatioTemporalTraceformer(
        loc_dim=2, d_model=32, n_heads=1, n_layers=4, d_ff=32, max_len=max_len, dropout=0
    ).to(device)

    model_time.set_log_transform(True)
    model_loc.set_log_transform(True)

    model_time.set_RN_train(False)
    model_time.set_AR_train(True)
    model_loc.set_RN_train(False)
    model_loc.set_AR_train(True)

    model_time.load_state_dict(torch.load(model_time_path, map_location=device))
    model_loc.load_state_dict(torch.load(model_loc_path,   map_location=device))

    model_time.eval()
    model_loc.eval()

    # ---------------------------
    # 1) learned full λ(t)
    # ---------------------------
    L = int(instance["length"])
    lam_all = []
    t_all = []
    events_all = []

    cur_t = 0.0
    for n in range(0, L):
        log_prob, cif, cif_input, delta = get_intensity_function_dopri5(
            model_time, instance, n=n, device=device
        )

        if hasattr(cif, "detach"):
            lam_piece = cif.detach().cpu().numpy()
        else:
            lam_piece = np.asarray(cif)

        if hasattr(cif_input, "detach"):
            t_piece = cif_input.detach().cpu().numpy()
        else:
            t_piece = np.asarray(cif_input)

        # shift to global time
        t_piece = t_piece + cur_t
        lam_all.append(lam_piece)
        t_all.append(t_piece)

        delta_val = float(delta.item()) if hasattr(delta, "item") else float(delta)
        cur_t += delta_val
        events_all.append(cur_t)

    t_learn = np.concatenate(t_all, axis=0)
    lam_learn = np.concatenate(lam_all, axis=0)

    # sort + deduplicate
    order = np.argsort(t_learn)
    t_learn = t_learn[order]
    lam_learn = lam_learn[order]
    t_learn, uniq_idx = np.unique(t_learn, return_index=True)
    lam_learn = lam_learn[uniq_idx]

    # ---------------------------
    # 2) rebuild GT events from instance
    # ---------------------------
    deltas = instance["deltas"]
    locs   = instance["locs"]

    deltas_np = deltas.detach().cpu().numpy() if hasattr(deltas, "detach") else np.asarray(deltas)
    locs_np   = locs.detach().cpu().numpy()   if hasattr(locs, "detach")   else np.asarray(locs)

    t_events = np.cumsum(deltas_np[:L], axis=0).astype(np.float64)
    gt_events = np.concatenate([t_events[:, None], locs_np[:L].astype(np.float64)], axis=1)

    # pick an interval n with relatively larger delta so t-slices show clearer change
    n_low = min(5, max(0, L - 1))
    n_high = max(n_low + 1, L)
    n_candidates = np.arange(n_low, n_high)
    if n_candidates.size > 0:
        n_vis = int(n_candidates[np.argmax(deltas_np[n_candidates])])
    else:
        n_vis = min(20, max(0, L - 1))
    print(f"Chosen n for t-x-y plots: n={n_vis}, delta={float(deltas_np[n_vis]):.6f}")

    # ---------------------------
    # 3) STSCP / AutoSTPP GT parameters
    # ---------------------------
    g0_cov = np.array([[0.25, 0.0], [0.0, 0.25]], dtype=np.float64)
    g2_cov = np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64)
    alpha = 0.4
    beta = 0.2
    mu = 1.0
    gamma = 2.0
    lamb_max = 10.0
    x_num = 51
    y_num = 51
    x_lim = (0.0, 1.0)
    y_lim = (0.0, 1.0)

    # sim_kwargs = dict(
    #     t_max=100.0,
    #     x_lim=(0.0, 1.0),
    #     y_lim=(0.0, 1.0),
    #     g0_cov=np.array([[0.25, 0.0], [0.0, 0.25]], dtype=np.float64),
    #     g2_cov=np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64),
    #     alpha=0.4,
    #     beta=0.2,
    #     mu=1.0,
    #     gamma=2.0,
    #     lamb_max=10.0,
    #     x_num=51,
    #     y_num=51,
    #     min_events=15,
    #     dt_bound=0.5,
    # )

    # ---------------------------
    # 4) GT λ(t|H) using STSCP
    # ---------------------------
    lam_gt = gt_lambda_t_given_history_stscp(
        t_learn,
        gt_events,
        alpha=alpha,
        beta=beta,
        mu=mu,
        gamma=gamma,
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
        lamb_max=lamb_max,
    )

    # ---------------------------
    # 5) plot learned vs GT λ(t)
    # ---------------------------
    figsize = (32, 6)
    dpi = 220
    event_target_count = 60

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(t_learn, lam_learn, alpha=0.85, linewidth=1.2, label="Learned $\\lambda(t)$")
    ax.plot(t_learn, lam_gt,    alpha=0.85, linewidth=1.2, linestyle="--", label="GT $\\lambda(t|H)$")

    ev = np.asarray(events_all, dtype=np.float64)
    if ev.size > 0:
        k = max(1, int(np.ceil(ev.size / event_target_count)))
        ev_plot = ev[::k]

        ymin, ymax = ax.get_ylim()
        rug_h = 0.06 * (ymax - ymin)
        ax.vlines(ev_plot, ymin, ymin + rug_h, alpha=0.35, linewidth=0.8)

    ax.set_title("STSCP: Learned vs Ground Truth $\\lambda(t)$ (full sequence)")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\lambda(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.margins(x=0)

    fig.tight_layout()
    fig.savefig("learned_vs_gt_SC.png")
    plt.show()
    plt.close(fig)

    print("Saved: learned_vs_gt.png")

    # ---------------------------
    # 6) t-x-y comparison using STSCP GT
    # ---------------------------
    plot_txy_compare(
        model_time=model_time,
        model_loc=model_loc,
        instance=instance,
        n=n_vis,
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        alpha=alpha,
        beta=beta,
        mu=mu,
        gamma=gamma,
        lamb_max=lamb_max,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
        grid_size=30,
        pick_k_times=3,
        device=device,
        out_prefix="learned_vs_gt_txy_SC",
    )

    # ---------------------------
    # 7) 3D t-x-y comparison using STSCP GT
    # ---------------------------
    # plot_txy_compare_3d(
    #     model_time=model_time,
    #     model_loc=model_loc,
    #     instance=instance,
    #     n=n_vis,
    #     g0_cov=g0_cov,
    #     g2_cov=g2_cov,
    #     alpha=alpha,
    #     beta=beta,
    #     mu=mu,
    #     gamma=gamma,
    #     lamb_max=lamb_max,
    #     x_num=x_num,
    #     y_num=y_num,
    #     x_lim=x_lim,
    #     y_lim=y_lim,
    #     grid_size=30,
    #     pick_k_times=4,
    #     sample_event_times=True,
    #     event_t_range=(50.0, 90.0),
    #     include_event_bracket=False,
    #     surface_cmap="YlGnBu",
    #     surface_alpha=0.88,
    #     elev=28,
    #     azim=-60,
    #     device=device,
    #     out_prefix="learned_vs_gt_txy_3d_SC",
    # )

    # ---------------------------
    # 8) Unified 5-row comparison:
    #    GT, ARCH, AutoSTPP, DeepSTPP, NSTPP
    #    Use the same sequence and the same picked times.
    # ---------------------------
    plot_sc_3d_with_baselines(
        model_time=model_time,
        model_loc=model_loc,
        seq_idx=n_vis,
        pick_k_times=5,
        time_range=(30.0, 100.0),
        baseline_root="Spatial_Temporal_Plotting",
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        alpha=alpha,
        beta=beta,
        mu=mu,
        gamma=gamma,
        lamb_max=lamb_max,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
        grid_size=30,
        surface_cmap="YlGnBu",
        surface_alpha=0.9,
        elev=28,
        azim=-60,
        device=device,
        out_file="stscp_3d_gt_arch_autostpp_deepstpp_nstpp.png",
    )

def visualize_learned_intensity_function(device='cuda', model_path='best_model.pth'):
    train_data, max_len_tr = load_data_3d(data_path('stscp_out/data_train.pkl'))
    test_data, max_len_te = load_data_3d(data_path('stscp_out/data_test.pkl'))
    max_len = max(max_len_tr, max_len_te)
    instance = test_data[20]  # visualize the first test instance
    model = SpatioTemporalTraceformer(
        loc_dim=2,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=128,
        max_len=max_len,
        dropout=0,
        loc_min=0.0,
        loc_max=1.0,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    #model.set_log_transform(False)
    model.set_log_transform(True)
    model.set_RN_train(False)
    model.set_AR_train(True)
    model.set_rescale(1.0)
    model.set_selection_combintation_weights(w_time_err=1.0, w_loc_err=1.0)


    model.eval()

    L = instance['length']
    cif_all = []
    cif_input_all = []
    events_all = []
    cur_delta = 0.0
    for n in range(0, L):
        _, cif, cif_input, delta = get_intensity_function(model, instance, n=n, device=device)
        cif_all.append(cif)
        cif_input += cur_delta
        cif_input_all.append(cif_input)
        cur_delta += delta
        events_all.append(cur_delta)
    cif = np.concatenate(cif_all, axis=0)
    cif_input = np.concatenate(cif_input_all, axis=0)
    import matplotlib.pyplot as plt
    plt.plot(cif_input, cif, color='blue', alpha=0.7, label='$\lambda(t)$ (Intensity)', linestyle='-')

    for event_time in events_all:
        plt.axvline(x=event_time, color='red', linestyle='--', alpha=0.3)

    plt.title(f'1D Hawkes Process Simulation via Learned Model')
    plt.xlabel('Time')
    plt.ylabel('CIF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig('learned_intensity_function.png')


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"Seed fixed to {seed}")

