
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
import copy

@dataclass
class STLossConfig:
    time_loss_weight: float = 1.0
    loc_loss_weight: float = 1.0
    rc_loss_weight: float = 1.0

class SpatioTemporalTraceformer(nn.Module):
    """
    Flow Matching ver Traceformer：
    - state z = [log(1+Δt), norm_loc_x, norm_loc_y] ∈ R^{1+loc_dim}
    - output velocity v_theta(z_s, s) ∈ R^{1+loc_dim}
    - train：FM loss (MSE on velocity)
    """

    def __init__(
            self,
            loc_dim: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            d_ff: int = 1024,
            max_len: int = 512,
            dropout: float = 0.1,
            d_time: int = 32,
            time_hidden: int = 64,
            loc_hidden: int = 64,
            delta_min: Optional[np.ndarray] = None,
            delta_max: Optional[np.ndarray] = None,
            loc_min: Optional[np.ndarray] = None,
            loc_max: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.loc_dim = loc_dim
        self.d_model = d_model
        self.d_time = d_time
        self.time_hidden = time_hidden
        self.loc_hidden = loc_hidden
        self.max_len = max_len

        if loc_min is not None and loc_max is not None:
            loc_min = torch.as_tensor(loc_min, dtype=torch.float32)
            loc_max = torch.as_tensor(loc_max, dtype=torch.float32)
            self.register_buffer("loc_min", loc_min)
            self.register_buffer("loc_max", loc_max)
        else:
            self.loc_min = None
            self.loc_max = None

        if delta_min is not None and delta_max is not None:
            delta_min = torch.as_tensor(delta_min, dtype=torch.float32)
            delta_max = torch.as_tensor(delta_max, dtype=torch.float32)
            self.register_buffer("delta_min", delta_min)
            self.register_buffer("delta_max", delta_max)
        else:
            self.delta_min = None
            self.delta_max = None

        # position encoding
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_emb", pe, persistent=True)
        self.drop = nn.Dropout(dropout)

        # ---------- time embedding ----------
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_time, self.time_hidden),
            nn.GELU(),
            nn.Linear(self.time_hidden, d_model),
        )

        # ---------- loc embedding ----------
        self.loc_mlp = nn.Sequential(
            nn.Linear(self.loc_dim, self.loc_hidden),
            nn.GELU(),
            nn.Linear(self.loc_hidden, d_model),
        )

        # ---------- flow time s embedding ----------
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(self.d_time, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ---------- History condition Encoder ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ---------- Transformer Decoder ----------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(d_model)

        # ---------- velocity Head ----------
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, 1 + loc_dim),
        )

        self.time_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, 1),
        )

        self.loc_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, self.loc_dim),
        )

        self.do_log_transform = True
        self.AR_training = True
        self.RN_training = True
        self.rescale = 1.0

        self.weight_time_error = 1.0
        self.weight_loc_error = 1.0
        self.weight_rc_error = 1.0


    def set_log_transform(self, do_log: bool):
        if not isinstance(do_log, bool):
            raise TypeError("do_log must be a bool")
        self.do_log_transform = do_log
        return self

    def set_AR_train(self, do_AR: bool):
        if not isinstance(do_AR, bool):
            raise TypeError("do_AR must be a bool")
        self.AR_training = do_AR
        return self

    def set_RN_train(self, do_RN: bool):
        if not isinstance(do_RN, bool):
            raise TypeError("do_RN must be a bool")
        self.RN_training = do_RN
        return self

    def continuous_timestep_embedding(self, t: torch.Tensor, max_period: float = 10000.0):
        t = t.float()
        half = self.d_time // 2
        max_period = torch.tensor(max_period, device=t.device, dtype=t.dtype)
        freqs = torch.exp(
            -torch.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )  # (half,)
        args = t[..., None] * freqs  # (..., half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., 2*half)
        if self.d_time % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def log_transform(self, deltas, eps: float = 1e-6):
        return torch.log(deltas + eps)

    def log_transform_inverse(self, deltas_log, eps: float = 1e-6):
        return deltas_log.exp() - eps

    def encode_history(
            self,
            deltas_hist: torch.Tensor,  # [B, T]
            locs_hist: torch.Tensor,  # [B, T, loc_dim]
            key_padding_mask: torch.Tensor,  # [B, T] true: this token is ignored
            attn_mask: torch.Tensor = None
            # [T, T] (i,j) means i's token cannot use j's token to compute attention (j cannot attend to i)
    ) -> torch.Tensor:
        B, T = deltas_hist.shape
        device = deltas_hist.device

        pos = torch.arange(T, device=device).clamp_max(self.max_len - 1)
        pos = torch.cat([torch.tensor([0], device=device), pos[0:-1]]) #the first token is padding
        pos_emb = self.pos_emb[pos]  # [T, d_model]

        time_emb = self.continuous_timestep_embedding(deltas_hist)  # [B, T, d_time]
        time_emb = self.time_mlp(time_emb)  # [B, T, d_model]

        if locs_hist is None:
            loc_emb = torch.zeros(B, T, self.d_model, device=device)
        else:
            loc_emb = self.loc_mlp(locs_hist)  # [B, T, d_model]

        x_hist = time_emb + loc_emb + pos_emb  # [B, T, d_model]
        x_hist = self.drop(x_hist)

        memory = self.encoder(
            x_hist,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,  # [B, T]
        )  # [B, T, d_model]

        return memory

    def forward(
            self,
            deltas_noisy: torch.Tensor,  # [B, T] time noisy state (z_s[...,0])
            locs_noisy: torch.Tensor,    # [B, T, loc_dim] loc noisy state (z_s[...,1:])
            flow_t: torch.Tensor,        # [B, T] flow time s
            deltas: torch.Tensor,
            locs: torch.Tensor,
            key_padding_mask: torch.Tensor,  # [B, T]
            deltas_mem: torch.Tensor,
            locs_mem: torch.Tensor,
            memory_key_padding_mask: torch.Tensor,  # [B, T]
            condition_mask: Optional[torch.Tensor] = None,  # [B, T]
            auto_regressive: bool = False,
            predict_loc_only: bool = False,
    ):
        B, T = deltas_noisy.shape
        device = deltas_noisy.device

        # ------- 1) encode real history condition -------
        if auto_regressive:
            idx = torch.arange(T, device=device)
            memory_attn_mask = idx[:, None] < idx[None, :]  # standard causal mask
            memory_key_padding = memory_key_padding_mask
            #memory_key_padding[:, 0] = False
        else:
            memory_attn_mask = torch.zeros(T, T, dtype=torch.bool, device=device)
            memory_attn_mask[0, 1:] = True   # row0: mask out attending to 1..T-1

            # memory_attn_mask = None  # full self-attention layers
            # memory_key_padding = ~((~key_padding_mask) & condition_mask)
            # full_padded_rows = torch.where(memory_key_padding.all(dim=1))[0]
            # memory_key_padding[
            #     full_padded_rows, -1] = False  # enforce the faked token can do self-attention & cross-attention

            if condition_mask is None:
                cond_mem = torch.ones_like(memory_key_padding_mask, dtype=torch.bool, device=device)
            else:
                cond_mem = torch.zeros_like(condition_mask, dtype=torch.bool, device=device)
                cond_mem[:, 1:] = condition_mask[:, :-1]
            memory_key_padding = ~( (~memory_key_padding_mask) & cond_mem )
            #memory_key_padding[:, 0] = False

        memory = self.encode_history(deltas_hist=deltas_mem, locs_hist=locs_mem, attn_mask=memory_attn_mask,
                                     key_padding_mask=memory_key_padding)  # [B, T, d_model]

        # ------- 2) decoder input：z_s + s -------
        pos = torch.arange(T, device=device).clamp_max(self.max_len - 1)
        pos_emb = self.pos_emb[pos]  # [T, d_model]

        # noisy state time embedding
        if predict_loc_only:
            time_for_emb = deltas  # [B, T]
        else:
            time_for_emb = deltas_noisy  # [B, T]

        time_emb = self.continuous_timestep_embedding(time_for_emb)  # [B, T, d_time]
        time_emb = self.time_mlp(time_emb)  # [B, T, d_model]

        # noisy state loc embedding
        if predict_loc_only is False: #time only
            loc_emb = torch.zeros(B, T, self.d_model, device=device)
        else:
            loc_emb = self.loc_mlp(locs_noisy)

        flow_emb = self.continuous_timestep_embedding(flow_t)  # [B, T, d_time]
        flow_emb = self.flow_time_mlp(flow_emb)  # [B, T, d_model]

        x = time_emb + loc_emb + pos_emb + flow_emb
        x = self.drop(x)  # [B, T, d_model]

        # ------- 3) real history condition causal mask -------
        # memory_mask: [T_tgt, T_src]，True is mask.
        # mask[i, j] = True  if j >= i  => mask self and future
        if auto_regressive:
            idx = torch.arange(T, device=device)
            memory_mask = idx[:, None] < idx[None, :]  # decoder i attend to memory 0..i
            memory_key_padding = memory_key_padding_mask.clone()
            memory_key_padding[:, 0] = False # ensure the first padding token in memory can be attended by decoder to avoid nan
            attn_mask = idx[:, None] < idx[None, :]  # self attention, token i can only attend to tokens 1..i
        else:
            memory_mask = None  # random conditon, decoder i can see every token in memory
            attn_mask = None  # random conditon, we generate all tokens together
            key_padding_mask = ~((~key_padding_mask) & (~condition_mask))  # mask out those to be fixed: an alternative option is not & conditon_mask, still generate all, but produce zero velocty for the fixed tokens
            full_padded_rows = torch.where(key_padding_mask.all(dim=1))[0]
            key_padding_mask[full_padded_rows, -1] = False  # enforce at least one token can do self-attention
            memory_key_padding = memory_key_padding
            memory_key_padding[:, 0] = False

        #memory = memory*0.0
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=attn_mask,
            memory_mask=memory_mask,  # [i,j] = true token-i in decoder cannot see token-j in encoder
            tgt_key_padding_mask=key_padding_mask, #tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding,
        )
        x = self.ln_out(x)  # [B, T, d_model]

        if predict_loc_only is True:
            v_time = torch.zeros(B, T, device=device)
            v_loc = self.loc_head(x)  # [B, T, loc_dim]
        elif predict_loc_only is False:
            v_time = self.time_head(x).squeeze(-1)  # [B, T]
            v_loc = torch.zeros(B, T, self.loc_dim, device=device)

        return v_time, v_loc

    def collate_with_mask(self, batch, max_len: Optional[int] = None):
        """Collate sequence dictionaries containing raw deltas [L], normalized locs [L, loc_dim], and noise. Return transformed deltas [B, T], locs [B, T, loc_dim], lengths [B], and predict_mask selecting indices 1..L-1."""
        B = len(batch)
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        T = int(lengths.max().item())
        if max_len and T > max_len:
            # T = max_len
            raise ValueError(f'seq length {T} > max-len {max_len}')

        device = batch[0]["deltas"].device

        #at least one padding in front of the memory to avoid nan in transformer attention
        deltas = torch.zeros(B, T + 1, dtype=torch.float, device=device)
        deltas_mem = torch.zeros(B, T + 1, dtype=torch.float, device=device)
        noise = torch.zeros(B, T + 1, dtype=torch.float, device=device)
        locs = torch.zeros(B, T + 1, self.loc_dim, dtype=torch.float, device=device)
        locs_mem = torch.zeros(B, T + 1, self.loc_dim, dtype=torch.float, device=device)
        predict_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        predict_mask_last = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        seq_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        mem_seq_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        noise_time = torch.zeros(B, T + 1, device=device)
        noise_loc  = torch.zeros(B, T + 1, self.loc_dim, device=device)


        for i, b in enumerate(batch):
            #decoder part
            L = b["length"]
            delta_vals = b["deltas"][:L].clone()
            if self.do_log_transform:
                delta_vals = self.log_transform(delta_vals)
            deltas[i, :L] = delta_vals
            #noise[i, :L] = b["noise"][:L]
            noise_time[i, :L] = b['noise_time'][:L]
            noise_loc[i, :L]  = b['noise_loc'][:L]
            locs[i, :L, :] = b["locs"][:L]
            predict_mask[i, 0:L] = True  #
            predict_mask_last[i, L - 1] = True  # only the last event
            seq_mask[i, :L] = True
            #memory part
            deltas_mem[i, 1:L + 1] = delta_vals  # memory has one padding at beginning
            locs_mem[i, 1:L + 1, :] = b["locs"][:L, :]  # memory has one padding at beginning
            mem_seq_mask[i, 1:L + 1] = True

        return {
            'deltas_tgt': deltas,  # [B, T]
            #'noise': noise,      # [B, T]
            'noise_time': noise_time,
            'noise_loc': noise_loc,
            'locs_tgt': locs,  # [B, T, loc_dim]
            'deltas_mem': deltas_mem,  # [B, T]
            'locs_mem': locs_mem,  # [B, T, loc_dim]
            'lengths': lengths.to(device),
            'seq_mask': seq_mask,
            'mem_seq_mask': mem_seq_mask,
            "predict_mask": predict_mask,  # mainly for test prediction
        }

    def compute_time_losses_flow(self, batch, loss_cfg: STLossConfig):
        device = next(self.parameters()).device

        deltas_tgt = batch["deltas_tgt"].to(device)  # [B, T]
        #noise = batch["noise"].to(device)  # [B, T]
        locs_tgt = batch["locs_tgt"].to(device)  # [B, T, loc_dim]
        predict_mask = batch["predict_mask"].to(device)
        seq_mask = batch["seq_mask"].to(device)

        #memory part: the first token is padding
        deltas_mem = batch["deltas_mem"].to(device)  # [B, T]
        locs_mem = batch["locs_mem"].to(device)  # [B, T, loc_dim]
        mem_seq_mask = batch["mem_seq_mask"].to(device)

        B, T = deltas_tgt.shape
        #z0 = noise
        z0 = batch['noise_time'].to(device)
        z1 = deltas_tgt
        s = torch.rand(B, T, device=device)  # [B, T]
        z_s = (1.0 - s) * z0 + s * z1  # [B, T]
        deltas_s = z_s
        locs_s = torch.zeros(B, T, self.loc_dim, device=device)  # zero loc part
        v_time_target = z1 - z0
        key_padding_mask = ~seq_mask  # [B, T]
        mem_key_padding = ~mem_seq_mask  # [B, T]
        t_loss_ar = torch.tensor(0.0, device=device)
        res_mask = None

        random_condition = batch.get("random_condition", None)
        if random_condition is None:
            random_condition = torch.rand_like(deltas_tgt) > 0.3
        random_condition = random_condition.to(device)

        t_loss_rn = torch.tensor(0.0, device=device)
        t_loss_ar = torch.tensor(0.0, device=device)

        if self.RN_training:
            v_t_pred, _ = self(
                deltas_noisy = deltas_s,
                locs_noisy = locs_s,
                flow_t = s,
                deltas = deltas_tgt,
                locs = locs_tgt,
                key_padding_mask = key_padding_mask,
                deltas_mem = deltas_mem,
                locs_mem = locs_mem,
                memory_key_padding_mask = mem_key_padding,
                condition_mask=random_condition,
                auto_regressive=False,
                predict_loc_only=False,
            )
            res_mask = seq_mask & ~random_condition
            if res_mask.any():
                t_loss_rn = (v_t_pred[res_mask] - v_time_target[res_mask]).square().mean()

        if self.AR_training:
            v_t_pred, _ = self(
                deltas_noisy = deltas_s,
                locs_noisy = locs_s,
                flow_t = s,
                deltas = deltas_tgt,
                locs = locs_tgt,
                key_padding_mask = key_padding_mask,
                deltas_mem = deltas_mem,
                locs_mem = locs_mem,
                memory_key_padding_mask = mem_key_padding,
                condition_mask=None,
                auto_regressive=True,
                predict_loc_only=False,
            )
            res_mask = predict_mask
            if res_mask.any():
                t_loss_ar = (v_t_pred[res_mask] - v_time_target[res_mask]).square().mean()

        loss = loss_cfg.time_loss_weight * (
                loss_cfg.rc_loss_weight * t_loss_rn + t_loss_ar
        )
        #loss = t_loss_ar

        metrics = {
            "loss": float(loss.item()),
            "mask_frac": float(res_mask.float().mean().item()) if res_mask is not None else 0.0,
        }
        return loss, metrics


    def compute_loc_losses_flow(self, batch, loss_cfg: STLossConfig):
        """Train spatial velocity with z1=locs_tgt, z0=noise_loc, z_s=(1-s)*z0+s*z1, and target velocity z1-z0. Condition the decoder on true deltas."""
        device = next(self.parameters()).device

        deltas_tgt = batch["deltas_tgt"].to(device)          # [B, T]
        locs_tgt   = batch["locs_tgt"].to(device)            # [B, T, loc_dim]
        seq_mask   = batch["seq_mask"].to(device)            # [B, T]
        predict_mask = batch["predict_mask"].to(device)      # [B, T]

        deltas_mem = batch["deltas_mem"].to(device)          # [B, T]
        locs_mem   = batch["locs_mem"].to(device)            # [B, T, loc_dim]
        mem_seq_mask = batch["mem_seq_mask"].to(device)      # [B, T]

        B, T = deltas_tgt.shape

        # -------- Flow Matching (loc) --------
        z1 = locs_tgt                                          # [B, T, loc_dim]
        z0 = batch["noise_loc"].to(device)                     # [B, T, loc_dim]
        s  = torch.rand(B, T, device=device)                   # [B, T]

        locs_s = (1.0 - s)[..., None] * z0 + s[..., None] * z1 # [B, T, loc_dim]
        v_loc_target = z1 - z0                                 # [B, T, loc_dim]

        deltas_s_dummy = deltas_tgt

        key_padding_mask = ~seq_mask
        mem_key_padding  = ~mem_seq_mask

        random_condition = batch.get("random_condition", None)
        if random_condition is None:
            random_condition = (torch.rand_like(deltas_tgt) > 0.3)
        random_condition = random_condition.to(device)

        loc_loss_rn = torch.tensor(0.0, device=device)
        loc_loss_ar = torch.tensor(0.0, device=device)
        res_mask = None

        # ---- RN ----
        if self.RN_training:
            _, v_loc_pred = self(
                deltas_noisy=deltas_s_dummy,
                locs_noisy=locs_s,
                flow_t=s,
                deltas=deltas_tgt,
                locs=locs_tgt,
                key_padding_mask=key_padding_mask,
                deltas_mem=deltas_mem,
                locs_mem=locs_mem,
                memory_key_padding_mask=mem_key_padding,
                condition_mask=random_condition,
                auto_regressive=False,
                predict_loc_only=True,
            )
            res_mask = seq_mask & ~random_condition
            if res_mask.any():
                loc_loss_rn = (v_loc_pred[res_mask] - v_loc_target[res_mask]).pow(2).sum(-1).mean()

        # ---- AR ----
        if self.AR_training:
            _, v_loc_pred = self(
                deltas_noisy=deltas_s_dummy,
                locs_noisy=locs_s,
                flow_t=s,
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
            res_mask = predict_mask
            if res_mask.any():
                loc_loss_ar = (v_loc_pred[res_mask] - v_loc_target[res_mask]).pow(2).sum(-1).mean()

        loss = loss_cfg.loc_loss_weight * (loss_cfg.rc_loss_weight * loc_loss_rn + loc_loss_ar)

        metrics = {
            "loss": float(loss.item()),
            "loc_loss_rn": float(loc_loss_rn.item()),
            "loc_loss_ar": float(loc_loss_ar.item()),
            "mask_frac": float(res_mask.float().mean().item()) if res_mask is not None else 0.0,
        }
        return loss, metrics

