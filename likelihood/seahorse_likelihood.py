from pathlib import Path
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "seahorse"

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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


LL_DATASET_PRESETS = {
    "earthquakes": {"display_name": "Earthquake", "time_unit": "day"},
    "covid19": {"display_name": "COVID19", "time_unit": "day"},
    "citibike": {"display_name": "Citibike", "time_unit": "hour"},
}


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


set_seed(42)


@dataclass
class STLossConfig:
    time_loss_weight: float = 1.3
    loc_loss_weight: float = 1.0
    rc_loss_weight: float = 1.0


class TimeLikelihood(Enum):
    LogNormalHomo = auto()
    LogNormal = auto()
    WeiBull = auto()


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
            time_head_hidden: int = 256,
            loc_head_hidden: int = 256,
            delta_min: Optional[np.ndarray] = None,
            delta_max: Optional[np.ndarray] = None,
            loc_min: Optional[np.ndarray] = None,
            loc_max: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.loc_dim = loc_dim
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

        self.d_model = d_model
        self.d_time = d_time
        self.time_hidden = time_hidden
        self.loc_hidden = loc_hidden
        self.max_len = max_len

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
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1 + loc_dim),
        )

        self.time_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.loc_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, self.loc_dim),
        )

        self._reset_parameters()

        self.do_log_transform = True
        self.AR_training = True
        self.RN_training = True
        self.delta_ll = TimeLikelihood.LogNormal
        self.rescale = 1.0

        self.weight_time_error = 1.0
        self.weight_loc_error = 1.0
        self.weight_rc_error = 1.0

    def set_selection_combintation_weights(self, w_time_err: float = 1.0, w_loc_err: float = 1.0, w_rc_err: float = 1.0):
        self.weight_time_error = w_time_err
        self.weight_loc_error = w_loc_err
        self.weight_rc_error = w_rc_err

    def set_rescale(self, rescale: float):
        if not isinstance(rescale, (int, float)):
            raise TypeError("rescale must be a numeric type")
        if rescale <= 0:
            raise ValueError("rescale must be positive")
        self.rescale = float(rescale)
        return self

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

    def set_time_likelihood(self, delta_ll):
        if isinstance(delta_ll, TimeLikelihood):
            self.delta_ll = delta_ll
            return self

        if not isinstance(delta_ll, str):
            raise TypeError("delta_ll must be a TimeLikelihood or str")
        key = delta_ll.strip().lower()
        if key in ("lognormal", "log_normal", "log-normal"):
            self.delta_ll = TimeLikelihood.LogNormal
        elif key in (
                "lognormalhomo",
                "log_normal_homo",
                "lognormal_homo",
                "lognormal-homo",
                "log-normal-homo",
                "homo",
                "homogeneous",
        ):
            self.delta_ll = TimeLikelihood.LogNormalHomo
        elif key in ("weibull", "weibull_dist", "weibull-distribution"):
            self.delta_ll = TimeLikelihood.WeiBull
        else:
            raise ValueError(f"Unknown delta_ll value: {delta_ll}")
        return self

    def _reset_parameters(self):
        for m in self.velocity_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.time_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.loc_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.loc_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.flow_time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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

    def log_transform(self, deltas):
        return torch.log(deltas + 1.0)

    def log_transform_inverse(self, deltas):
        return deltas.exp() - 1.0

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
        pos_emb = self.pos_emb[pos]  # [T, d_model]

        time_emb = self.continuous_timestep_embedding(deltas_hist)  # [B, T, d_time]
        time_emb = self.time_mlp(time_emb)  # [B, T, d_model]

        loc_emb = self.loc_mlp(locs_hist)  # [B, T, d_model]

        x_hist = time_emb + loc_emb + pos_emb  # [B, T, d_model]
        x_hist = self.drop(x_hist)

        # src_mask = make_causal_mask(T, device)  # [T, T]，True mask future

        memory = self.encoder(
            x_hist,
            mask=attn_mask,
            # mask=src_mask,
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
            condition_mask: torch.Tensor,    # [B, T] or None
            auto_regressive: bool = False,
            predict_loc_only: bool = False,
    ):
        B, T = deltas_noisy.shape
        device = deltas_noisy.device

        # ------- 1) encode real history condition -------
        if auto_regressive:
            idx = torch.arange(T, device=device)
            memory_attn_mask = idx[:, None] < idx[None, :]  # causal mask
            memory_key_padding = key_padding_mask
        else:
            memory_attn_mask = None  # full self-attention layers
            memory_key_padding = ~((~key_padding_mask) & condition_mask)
            full_padded_rows = torch.where(memory_key_padding.all(dim=1))[0]
            memory_key_padding[full_padded_rows, 0] = False

        memory = self.encode_history(deltas, locs, attn_mask=memory_attn_mask,
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
        if predict_loc_only is False:
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
            memory_mask = idx[:, None] <= idx[None, :]  # True means masked (j >= i)
            memory_mask[0, 0] = False                   # row0 must attend to dummy to avoid all-True
            memory_key_padding[:, 0] = False            # dummy never masked (extra safety)
            attn_mask = idx[:, None] < idx[None, :]     # causal self-attn
        else:
            memory_mask = None  # random conditon, decoder i can see every token in memory
            attn_mask = None  # random conditon, we generate all tokens together
            key_padding_mask = ~((~key_padding_mask) & (
                ~condition_mask))  # mask out those to be fixed: an alternative option is not & conditon_mask, still generate all, but produce zero velocty for the fixed tokens
            full_padded_rows = torch.where(key_padding_mask.all(dim=1))[0]
            key_padding_mask[full_padded_rows, 0] = False  # enforce at least one token can do self-attention

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=attn_mask,
            memory_mask=memory_mask,  # [i,j] = true token-i in decoder cannot see token-j in encoder
            tgt_key_padding_mask=key_padding_mask,
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
        """Prepend an all-zero dummy at index 0. Real events occupy 1..L; predict_mask scores 2..L, excluding the dummy and first real event."""
        B = len(batch)
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        T_real = int(lengths.max().item())

        if max_len and T_real + 1 > max_len:
            raise ValueError(f"seq length {T_real+1} > max-len {max_len}")

        device = batch[0]["deltas"].device

        # total length = real length + 1 leading dummy
        T = T_real + 1

        deltas = torch.zeros(B, T, dtype=torch.float, device=device)
        locs = torch.zeros(B, T, self.loc_dim, dtype=torch.float, device=device)
        predict_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        for i, b in enumerate(batch):
            L = int(b["length"])
            delta_vals = b["deltas"][:L].clone()

            if self.do_log_transform:
                delta_vals = self.rescale * delta_vals
                delta_vals = self.log_transform(delta_vals)

            # shift by +1: put real tokens at 1..L
            if L > 0:
                deltas[i, 1:L+1] = delta_vals
                locs[i, 1:L+1, :] = b["locs"][:L]

            # dummy + real tokens are valid
            seq_mask[i, :L+1] = True  # includes dummy at 0

            # teacher-style: predict tokens that have >=1 real history
            # (skip dummy(0) and first real event(1))
            if L >= 2:
                predict_mask[i, 2:L+1] = True

        return {
            "deltas_tgt": deltas,
            "locs_tgt": locs,
            "lengths": lengths.to(device),
            "seq_mask": seq_mask,
            "predict_mask": predict_mask,
        }

    def compute_time_losses_flow(self, batch, loss_cfg: STLossConfig):
        """Train the temporal velocity with flow matching and the configured random-condition/autoregressive masks; exclude spatial velocity from this loss."""
        device = next(self.parameters()).device

        deltas_tgt = batch["deltas_tgt"].to(device)
        locs_tgt = batch["locs_tgt"].to(device)  # [B, T, loc_dim]
        predict_mask = batch["predict_mask"].to(device)
        seq_mask = batch["seq_mask"].to(device)

        B, T = deltas_tgt.shape

        z1 = torch.cat(
            [deltas_tgt.unsqueeze(-1), locs_tgt], dim=-1
        )  # [B, T, 1+loc_dim]
        z0 = torch.randn_like(z1)  # [B, T, 1+loc_dim]
        s = torch.rand(B, T, device=device)  # [B, T]

        z_s = (1.0 - s)[..., None] * z0 + s[..., None] * z1  # [B, T, 1+loc_dim]
        deltas_s = z_s[..., 0]  # [B, T]
        locs_s = z_s[..., 1:]  # [B, T, loc_dim]

        v_target = z1 - z0  # [B, T, 1+loc_dim]
        v_time_target = v_target[..., 0]  # [B, T]

        key_padding_mask = ~seq_mask  # [B, T]

        random_condition = batch.get("random_condition", None)
        if random_condition is None:
            random_condition = torch.rand_like(deltas_tgt) > 0.3
        random_condition = random_condition.to(device)
        random_condition = random_condition & seq_mask
        random_condition[:, 0] = True

        t_loss_rn = torch.tensor(0.0, device=device)
        t_loss_ar = torch.tensor(0.0, device=device)
        res_mask = None

        if self.RN_training:
            v_t_pred, _ = self(
                deltas_s,
                locs_s,
                s,
                deltas_tgt,
                locs_tgt,
                key_padding_mask,
                random_condition,
                auto_regressive=False,
                predict_loc_only=False,
            )
            res_mask = seq_mask & ~random_condition
            if res_mask.any():
                t_loss_rn = (v_t_pred[res_mask] - v_time_target[res_mask]).square().mean()

        if self.AR_training:
            v_t_pred, _ = self(
                deltas_s,
                locs_s,
                s,
                deltas_tgt,
                locs_tgt,
                key_padding_mask,
                None,
                auto_regressive=True,
                predict_loc_only=False,
            )
            res_mask = predict_mask
            if res_mask.any():
                t_loss_ar = (v_t_pred[res_mask] - v_time_target[res_mask]).square().mean()

        loss = loss_cfg.time_loss_weight * (
                loss_cfg.rc_loss_weight * t_loss_rn + t_loss_ar
        )

        metrics = {
            "loss": float(loss.item()),
            "time_loss": float((t_loss_ar + t_loss_rn).item()),
            "mask_frac": float(res_mask.float().mean().item()) if res_mask is not None else 0.0,
        }
        return loss, metrics

    def compute_loc_losses_flow(self, batch, loss_cfg: STLossConfig):
        """Train the spatial velocity with flow matching. The decoder conditions on true time during training and predicted time during sampling."""
        device = next(self.parameters()).device

        deltas_tgt = batch["deltas_tgt"].to(device)  # [B, T]
        locs_tgt = batch["locs_tgt"].to(device)  # [B, T, loc_dim]
        predict_mask = batch["predict_mask"].to(device)
        seq_mask = batch["seq_mask"].to(device)

        B, T = deltas_tgt.shape

        z1 = torch.cat(
            [deltas_tgt.unsqueeze(-1), locs_tgt], dim=-1
        )  # [B, T, 1+loc_dim]
        z0 = torch.randn_like(z1)  # [B, T, 1+loc_dim]
        s = torch.rand(B, T, device=device)  # [B, T]

        z_s = (1.0 - s)[..., None] * z0 + s[..., None] * z1  # [B, T, 1+loc_dim]
        deltas_s = z_s[..., 0]  # [B, T]
        locs_s = z_s[..., 1:]  # [B, T, loc_dim]

        v_target = z1 - z0
        v_loc_target = v_target[..., 1:]  # [B, T, loc_dim]

        key_padding_mask = ~seq_mask

        random_condition = batch.get("random_condition", None)
        if random_condition is None:
            random_condition = torch.rand_like(deltas_tgt) > 0.3
        random_condition = random_condition.to(device)
        random_condition = random_condition & seq_mask
        random_condition[:, 0] = True

        loc_loss_rn = torch.tensor(0.0, device=device)
        loc_loss_ar = torch.tensor(0.0, device=device)
        res_mask = None

        if self.RN_training:
            _, v_loc_pred = self(
                deltas_s,
                locs_s,
                s,
                deltas_tgt,
                locs_tgt,
                key_padding_mask,
                random_condition,
                auto_regressive=False,
                predict_loc_only=True,
            )
            res_mask = seq_mask & ~random_condition
            if res_mask.any():
                loc_loss_rn = (v_loc_pred[res_mask] - v_loc_target[res_mask]).pow(2).sum(-1).mean()

        if self.AR_training:
            _, v_loc_pred = self(
                deltas_s,
                locs_s,
                s,
                deltas_tgt,
                locs_tgt,
                key_padding_mask,
                None,
                auto_regressive=True,
                predict_loc_only=True,
            )
            res_mask = predict_mask
            if res_mask.any():
                loc_loss_ar = (v_loc_pred[res_mask] - v_loc_target[res_mask]).pow(2).sum(-1).mean()

        loss = loss_cfg.loc_loss_weight * (
                loss_cfg.rc_loss_weight * loc_loss_rn + loc_loss_ar
        )

        metrics = {
            "loss": float(loss.item()),
            "loc_loss": float((loc_loss_ar + loc_loss_rn).item()),
            "mask_frac": float(res_mask.float().mean().item()) if res_mask is not None else 0.0,
        }
        return loss, metrics


def eval_two_stage(
        model_time: SpatioTemporalTraceformer,
        model_loc: SpatioTemporalTraceformer,
        dl: torch.utils.data.DataLoader,
        loss_cfg: STLossConfig,
):
    device = next(model_time.parameters()).device
    model_time.eval()
    model_loc.eval()

    total_loss = 0.0
    total_time_loss = 0.0
    total_loc_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dl:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            B = batch["deltas_tgt"].shape[0]
            #batch["random_condition"] = (torch.rand_like(batch["deltas_tgt"]) > 0.3).to(device)
            rc = (torch.rand_like(batch["deltas_tgt"]) > 0.3).to(device)
            rc = rc & batch["seq_mask"].to(device)
            rc[:, 0] = True
            batch["random_condition"] = rc

            time_loss, t_metrics = model_time.compute_time_losses_flow(batch, loss_cfg)
            loc_loss,  l_metrics = model_loc.compute_loc_losses_flow(batch, loss_cfg)
            loss = time_loss + loc_loss

            total_loss      += loss.item() * B
            total_time_loss += t_metrics["time_loss"] * B
            total_loc_loss  += l_metrics["loc_loss"] * B
            total_count     += B

    if total_count == 0:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / total_count
    avg_time_loss = total_time_loss / total_count
    avg_loc_loss  = total_loc_loss  / total_count
    return avg_loss, avg_time_loss, avg_loc_loss

def get_two_stage_prediction_error_sampled(
        model_time: SpatioTemporalTraceformer,
        model_loc: SpatioTemporalTraceformer,
        dl: torch.utils.data.DataLoader,
        num_steps_time: int = 100,
        num_steps_loc: int = 100,
        num_run: int = 1,
):
    """Integrate time from zero noise, then integrate location conditioned on predicted time. Return time RMSE and mean spatial distance in original units."""
    device = next(model_time.parameters()).device
    model_time.eval()
    model_loc.eval()

    set_seed(42+num_run)

    delta_pred_list = []
    delta_gt_list   = []
    loc_pred_list   = []
    loc_gt_list     = []

    with torch.no_grad():
        for batch in dl:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            deltas_tgt   = batch["deltas_tgt"]
            locs_tgt     = batch["locs_tgt"]     # [B, T, loc_dim]
            seq_mask     = batch["seq_mask"]     # [B, T]
            predict_mask = batch["predict_mask"] # [B, T]

            B, T = deltas_tgt.shape
            loc_dim = model_time.loc_dim

            if not predict_mask.any():
                continue

            key_padding_mask = ~seq_mask

            z_time = torch.zeros(B, T, 1, device=device)  # [B, T, 1]
            s_val  = 0.0
            ds_time = 1.0 / num_steps_time

            for _ in range(num_steps_time):
                s = torch.full((B, T), s_val, device=device)  # [B, T]

                deltas_s      = z_time[..., 0]                               # [B, T]
                locs_s_dummy  = torch.zeros(B, T, loc_dim, device=device)    # [B, T, loc_dim]

                v_time, _ = model_time(
                    deltas_s,
                    locs_s_dummy,
                    s,
                    deltas_tgt,
                    locs_tgt,
                    key_padding_mask,
                    None,
                    auto_regressive=True,
                    predict_loc_only=False,
                )

                v = v_time.unsqueeze(-1)           # [B, T, 1]
                z_time = z_time + ds_time * v
                s_val += ds_time

            delta_pred_log_full = z_time[..., 0]  # [B, T]

            mask = predict_mask
            delta_gt_log = deltas_tgt[mask]           # [N_all]
            delta_pred_log = delta_pred_log_full[mask]

            if model_time.do_log_transform:
                delta_gt_real   = model_time.log_transform_inverse(delta_gt_log)   / model_time.rescale
                delta_pred_real = model_time.log_transform_inverse(delta_pred_log) / model_time.rescale
            else:
                delta_gt_real   = delta_gt_log   / model_time.rescale
                delta_pred_real = delta_pred_log / model_time.rescale

            if model_time.delta_min is not None and model_time.delta_max is not None:
                delta_min = model_time.delta_min.to(device)
                delta_max = model_time.delta_max.to(device)
                scale = (delta_max - delta_min)
                delta_gt_real   = delta_gt_real   * scale + delta_min
                delta_pred_real = delta_pred_real * scale + delta_min

            delta_pred_list.append(delta_pred_real.cpu())
            delta_gt_list.append(delta_gt_real.cpu())

            z_loc = torch.zeros(B, T, loc_dim, device=device)  # [B, T, loc_dim]
            s_val = 0.0
            ds_loc = 1.0 / num_steps_loc

            for _ in range(num_steps_loc):
                s = torch.full((B, T), s_val, device=device)  # [B, T]

                deltas_noisy_for_loc = delta_pred_log_full  # [B, T]

                _, v_loc = model_loc(
                    deltas_noisy_for_loc,
                    z_loc,
                    s,
                    delta_pred_log_full,
                    locs_tgt,
                    key_padding_mask,
                    None,
                    auto_regressive=True,
                    predict_loc_only=True,
                )

                z_loc = z_loc + ds_loc * v_loc
                s_val += ds_loc

            loc_pred_norm_full = z_loc      # [B, T, loc_dim]
            loc_gt_norm_full   = locs_tgt   # [B, T, loc_dim]

            loc_pred_norm = loc_pred_norm_full[mask]  # [N_all, loc_dim]
            loc_gt_norm   = loc_gt_norm_full[mask]

            if model_loc.loc_min is not None and model_loc.loc_max is not None:
                loc_min = model_loc.loc_min.to(device)
                loc_max = model_loc.loc_max.to(device)
                scale = (loc_max - loc_min) + 1e-8
                loc_pred_real = loc_pred_norm * scale + loc_min
                loc_gt_real   = loc_gt_norm   * scale + loc_min
            else:
                loc_pred_real = loc_pred_norm
                loc_gt_real   = loc_gt_norm

            loc_pred_list.append(loc_pred_real.cpu())
            loc_gt_list.append(loc_gt_real.cpu())

    if not delta_pred_list:
        return float("nan"), float("nan")

    delta_pred_all = torch.cat(delta_pred_list, dim=0)
    delta_gt_all   = torch.cat(delta_gt_list,   dim=0)
    time_rmse = torch.sqrt(torch.mean((delta_pred_all - delta_gt_all) ** 2))

    loc_pred_all = torch.cat(loc_pred_list, dim=0)
    loc_gt_all   = torch.cat(loc_gt_list,   dim=0)
    loc_dists = torch.sqrt(torch.sum((loc_pred_all - loc_gt_all) ** 2, dim=-1))
    loc_euclid = loc_dists.mean()

    return time_rmse.item(), loc_euclid.item()


def _build_likelihood_prefix_batch(
        model: SpatioTemporalTraceformer,
        instance: dict,
        query_indices: torch.Tensor,
        device: torch.device,
):
    """
    Build one row per queried real event.

    Token 0 is the same leading dummy used during training.  Real event q is
    token q+1.  The model's AR cross-attention mask hides memory position i
    from decoder query i, so the queried event cannot leak into its history.
    """
    if query_indices.numel() == 0:
        raise ValueError("query_indices must not be empty")

    query_indices = query_indices.to(dtype=torch.long, device="cpu")
    event_positions = query_indices + 1
    batch_size = int(query_indices.numel())
    token_count = int(event_positions.max().item()) + 1
    loc_dim = int(instance["locs"].shape[-1])

    raw_deltas = instance["deltas"][: token_count - 1].to(device)
    delta_model = raw_deltas * model.rescale
    if model.do_log_transform:
        delta_values = model.log_transform(delta_model)
    else:
        delta_values = delta_model
    locations = instance["locs"][: token_count - 1].to(device)

    deltas_tgt = torch.zeros(batch_size, token_count, device=device)
    locs_tgt = torch.zeros(batch_size, token_count, loc_dim, device=device)
    seq_mask = torch.zeros(
        batch_size, token_count, dtype=torch.bool, device=device
    )

    for row, event_position in enumerate(event_positions.tolist()):
        # Includes dummy 0 and real events 1..event_position.
        deltas_tgt[row, 1 : event_position + 1] = delta_values[:event_position]
        locs_tgt[row, 1 : event_position + 1] = locations[:event_position]
        seq_mask[row, : event_position + 1] = True

    return {
        "deltas_tgt": deltas_tgt,
        "locs_tgt": locs_tgt,
        "key_padding_mask": ~seq_mask,
        "event_positions": event_positions.to(device),
        "query_indices": query_indices,
    }


def _time_log_probability_batch(
        model_time: SpatioTemporalTraceformer,
        instance: dict,
        query_indices: torch.Tensor,
        device: torch.device,
        rtol: float,
        atol: float,
):
    batch = _build_likelihood_prefix_batch(
        model_time, instance, query_indices, device
    )
    event_positions = batch["event_positions"]
    rows = torch.arange(event_positions.numel(), device=device)
    batch_size, token_count = batch["deltas_tgt"].shape

    raw_delta = instance["deltas"][query_indices].to(device)
    delta_model = raw_delta * model_time.rescale
    if model_time.do_log_transform:
        x1 = model_time.log_transform(delta_model)
    else:
        x1 = delta_model

    locs_dummy = torch.zeros(
        batch_size, token_count, model_time.loc_dim, device=device
    )

    class TimeODE(nn.Module):
        def forward(self, flow_time, state):
            x, accumulated_divergence = state
            del accumulated_divergence
            x = x.detach().requires_grad_(True)
            deltas_noisy = batch["deltas_tgt"].clone()
            deltas_noisy[rows, event_positions] = x
            flow = torch.ones_like(deltas_noisy) * flow_time

            velocity, _ = model_time(
                deltas_noisy=deltas_noisy,
                locs_noisy=locs_dummy,
                flow_t=flow,
                deltas=batch["deltas_tgt"],
                locs=batch["locs_tgt"],
                key_padding_mask=batch["key_padding_mask"],
                condition_mask=None,
                auto_regressive=True,
                predict_loc_only=False,
            )
            velocity = velocity[rows, event_positions]
            divergence = torch.autograd.grad(
                velocity.sum(),
                x,
                create_graph=False,
                retain_graph=False,
            )[0]
            return velocity.detach(), divergence.detach()

    zero = torch.zeros_like(x1)
    integration_times = torch.tensor(
        [1.0, 0.0], dtype=x1.dtype, device=device
    )
    x_path, divergence_path = odeint(
        TimeODE(),
        (x1, zero),
        integration_times,
        method="dopri5",
        rtol=rtol,
        atol=atol,
        options={"max_num_steps": 500},
    )
    x0 = x_path[-1]
    log_probability_model = (
        -0.5 * math.log(2.0 * math.pi)
        - 0.5 * x0.square()
        + divergence_path[-1]
    )

    # This model trains on y=log(1 + rescale*delta_raw), not on
    # log(delta+EPS) as in the EventFM_zhe likelihood script.
    if model_time.do_log_transform:
        log_abs_jacobian = (
            math.log(model_time.rescale) - torch.log1p(delta_model)
        )
    else:
        log_abs_jacobian = torch.full_like(
            delta_model, math.log(model_time.rescale)
        )
    return log_probability_model + log_abs_jacobian


def _location_log_probability_batch(
        model_loc: SpatioTemporalTraceformer,
        instance: dict,
        query_indices: torch.Tensor,
        device: torch.device,
        rtol: float,
        atol: float,
):
    batch = _build_likelihood_prefix_batch(
        model_loc, instance, query_indices, device
    )
    event_positions = batch["event_positions"]
    rows = torch.arange(event_positions.numel(), device=device)
    batch_size, token_count = batch["deltas_tgt"].shape
    loc_dim = model_loc.loc_dim
    x1 = instance["locs"][query_indices].to(device)

    # Defensive removal of the query location from the conditioning tensor.
    # The AR cross-attention mask already hides this token.
    locs_condition = batch["locs_tgt"].clone()
    locs_condition[rows, event_positions] = 0.0

    class LocationODE(nn.Module):
        def forward(self, flow_time, state):
            x, accumulated_divergence = state
            del accumulated_divergence
            x = x.detach().requires_grad_(True)
            locations_noisy = locs_condition.clone()
            locations_noisy[rows, event_positions] = x
            flow = torch.ones(
                batch_size,
                token_count,
                dtype=x.dtype,
                device=device,
            ) * flow_time

            _, velocity = model_loc(
                deltas_noisy=batch["deltas_tgt"],
                locs_noisy=locations_noisy,
                flow_t=flow,
                deltas=batch["deltas_tgt"],
                locs=locs_condition,
                key_padding_mask=batch["key_padding_mask"],
                condition_mask=None,
                auto_regressive=True,
                predict_loc_only=True,
            )
            velocity = velocity[rows, event_positions]
            diagonal_terms = []
            for dimension in range(loc_dim):
                gradient = torch.autograd.grad(
                    velocity[:, dimension].sum(),
                    x,
                    retain_graph=dimension + 1 < loc_dim,
                    create_graph=False,
                )[0]
                diagonal_terms.append(gradient[:, dimension])
            divergence = torch.stack(diagonal_terms, dim=0).sum(dim=0)
            return velocity.detach(), divergence.detach()

    zero = torch.zeros(batch_size, dtype=x1.dtype, device=device)
    integration_times = torch.tensor(
        [1.0, 0.0], dtype=x1.dtype, device=device
    )
    x_path, divergence_path = odeint(
        LocationODE(),
        (x1, zero),
        integration_times,
        method="dopri5",
        rtol=rtol,
        atol=atol,
        options={"max_num_steps": 500},
    )
    x0 = x_path[-1]
    return (
        -0.5 * loc_dim * math.log(2.0 * math.pi)
        - 0.5 * x0.square().sum(dim=-1)
        + divergence_path[-1]
    )


def _sequence_log_probabilities(
        model_time: SpatioTemporalTraceformer,
        model_loc: SpatioTemporalTraceformer,
        instance: dict,
        device: torch.device,
        event_batch_size: int,
        rtol: float,
        atol: float,
):
    time_parts = []
    location_parts = []
    length = int(instance["length"])
    # query index 0 is the first real event. It is excluded because its delta is
    # a loader sentinel and it is not part of the AR training objective.
    first_query_index = 0
    for start in range(first_query_index, length, event_batch_size):
        query_indices = torch.arange(
            start,
            min(start + event_batch_size, length),
            dtype=torch.long,
        )
        time_parts.append(
            _time_log_probability_batch(
                model_time,
                instance,
                query_indices,
                device,
                rtol,
                atol,
            ).detach().cpu()
        )
        location_parts.append(
            _location_log_probability_batch(
                model_loc,
                instance,
                query_indices,
                device,
                rtol,
                atol,
            ).detach().cpu()
        )
    if not time_parts:
        return torch.empty(0), torch.empty(0)
    return torch.cat(time_parts), torch.cat(location_parts)


def dataset_average_log_likelihood(
        model_time: SpatioTemporalTraceformer,
        model_loc: SpatioTemporalTraceformer,
        data,
        dataset_name: str,
        split_name: str,
        time_unit: str,
        device: torch.device,
        event_batch_size: int = 64,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-5,
        progress_every: int = 10,
):
    """
    Event-weighted conditional density from the second real event onward.
    The first event is excluded because its delta is a loader sentinel and is
    not trained by the AR objective. Terminal survival is omitted.
    """
    if model_time.delta_min is not None or model_time.delta_max is not None:
        raise NotImplementedError(
            "Likelihood currently assumes the unchanged training pipeline, "
            "which does not delta-min/max normalize inputs."
        )
    if model_loc.loc_min is None or model_loc.loc_max is None:
        raise ValueError("Raw-location likelihood requires loc_min and loc_max.")

    model_time.eval()
    model_loc.eval()
    original_requires_grad = []
    for model in (model_time, model_loc):
        for parameter in model.parameters():
            original_requires_grad.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)

    expected_events = sum(
        int(instance["length"]) for instance in data
    )
    time_sum = 0.0
    location_normalized_sum = 0.0
    total_events = 0
    try:
        with torch.enable_grad():
            for sequence_index, instance in enumerate(data):
                log_time, log_location = _sequence_log_probabilities(
                    model_time,
                    model_loc,
                    instance,
                    device,
                    event_batch_size,
                    rtol,
                    atol,
                )
                if not torch.isfinite(log_time).all():
                    raise FloatingPointError(
                        "Non-finite time likelihood in sequence "
                        f"{sequence_index}."
                    )
                if not torch.isfinite(log_location).all():
                    raise FloatingPointError(
                        "Non-finite location likelihood in sequence "
                        f"{sequence_index}."
                    )
                time_sum += float(log_time.double().sum())
                location_normalized_sum += float(log_location.double().sum())
                total_events += int(log_time.numel())

                if progress_every > 0 and (
                    (sequence_index + 1) % progress_every == 0
                    or sequence_index + 1 == len(data)
                ):
                    print(
                        f"[LL:{dataset_name}/{split_name}] sequences "
                        f"{sequence_index + 1}/{len(data)}, "
                        f"events={total_events}"
                    )
    finally:
        for parameter, required_grad in original_requires_grad:
            parameter.requires_grad_(required_grad)

    if total_events == 0:
        raise ValueError("Cannot average likelihood over zero events.")
    if total_events != expected_events:
        raise AssertionError(
            f"Scored {total_events} events, expected {expected_events}."
        )

    location_scale = (
        model_loc.loc_max.detach().double()
        - model_loc.loc_min.detach().double()
        + 1.0e-8
    )
    location_correction = -float(torch.log(location_scale).sum())
    avg_time = time_sum / total_events
    avg_location_normalized = location_normalized_sum / total_events
    avg_location_raw = avg_location_normalized + location_correction

    result = {
        "dataset": dataset_name,
        "split": split_name,
        "sequences": len(data),
        "events": total_events,
        "includes_first_event": False,
        "first_scored_event": "second real event",
        "first_event_excluded_reason": (
            "loader delta is a 1e-6 sentinel and token is not in AR loss"
        ),
        "terminal_survival_included": False,
        "averaging": (
            "sum from the second event onward / total scored events"
        ),
        "time_unit": time_unit,
        "location_raw_unit": "longitude_degree x latitude_degree",
        "time_scale_used_by_model": model_time.rescale,
        "time_transform": "log(1 + time_scale * delta_raw)",
        "location_normalizer_scope": (
            "unchanged utils.load_earthquake_st_ori behavior (all splits)"
        ),
        "avg_loglik_time": avg_time,
        f"avg_loglik_time_per_{time_unit}": avg_time,
        "avg_loglik_location_normalized": avg_location_normalized,
        "avg_loglik_joint_normalized": avg_time + avg_location_normalized,
        "location_jacobian_correction_to_raw": location_correction,
        "avg_loglik_location_raw_lon_lat": avg_location_raw,
        "avg_loglik_joint_raw_lon_lat": avg_time + avg_location_raw,
        "rtol": rtol,
        "atol": atol,
    }
    return result


def _fixed_validation_subset(data, max_sequences: Optional[int]):
    """Select a deterministic, evenly spaced validation subset for LL."""
    if max_sequences is None or max_sequences >= len(data):
        return data
    if max_sequences <= 0:
        raise ValueError("ll_checkpoint_max_sequences must be positive or None.")
    indices = np.linspace(
        0,
        len(data) - 1,
        num=max_sequences,
        dtype=np.int64,
    )
    return [data[int(index)] for index in indices]


def run_two_stage(
        model_time: SpatioTemporalTraceformer,
        model_loc: SpatioTemporalTraceformer,
        train_data,
        val_data,
        test_data,
        max_len,
        learning_rate=1e-3,
        max_epochs=93,
        batch_size_train=32,
        batch_size_eval=256,
        num_run=1,
        ll_dataset_name: str = "dataset",
        ll_time_unit: str = "time_unit",
        ll_checkpoint_every: int = 25,
        ll_checkpoint_max_sequences: Optional[int] = 32,
        ll_checkpoint_event_batch_size: int = 64,
        ll_checkpoint_rtol: float = 5.0e-4,
        ll_checkpoint_atol: float = 5.0e-5,
):
    device = next(model_time.parameters()).device
    assert next(model_loc.parameters()).device == device

    dl_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=lambda b: model_time.collate_with_mask(b, max_len),
    )
    dl_valid = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=lambda b: model_time.collate_with_mask(b, max_len),
    )
    dl_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=lambda b: model_time.collate_with_mask(b, max_len),
    )

    opt_time = torch.optim.AdamW(model_time.parameters(), lr=learning_rate)
    opt_loc  = torch.optim.AdamW(model_loc.parameters(),  lr=learning_rate)

    loss_cfg = STLossConfig(time_loss_weight=0.5, loc_loss_weight=1.0, rc_loss_weight=1.5)

    if ll_checkpoint_every <= 0:
        raise ValueError("ll_checkpoint_every must be positive.")
    validation_ll_data = _fixed_validation_subset(
        val_data, ll_checkpoint_max_sequences
    )
    best_val_joint_ll = -float("inf")
    best_val_ll_result = None
    best_ll_epoch = None
    best_time_state = None
    best_loc_state = None

    for epoch in range(max_epochs):
        model_time.train()
        model_loc.train()

        epoch_loss = 0.0
        epoch_time_loss = 0.0
        epoch_loc_loss = 0.0
        n_batch = 0

        for step, batch in enumerate(dl_train, start=1):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            batch["random_condition"] = (torch.rand_like(batch["deltas_tgt"]) > 0.3).to(device)

            time_loss, t_metrics = model_time.compute_time_losses_flow(batch, loss_cfg)
            loc_loss,  l_metrics = model_loc.compute_loc_losses_flow(batch, loss_cfg)
            loss = time_loss + loc_loss

            opt_time.zero_grad(set_to_none=True)
            opt_loc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_time.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_loc.parameters(),  max_norm=1.0)
            opt_time.step()
            opt_loc.step()

            epoch_loss      += loss.item()
            epoch_time_loss += t_metrics["time_loss"]
            epoch_loc_loss  += l_metrics["loc_loss"]
            n_batch += 1

        train_loss      = epoch_loss / max(1, n_batch)
        train_time_loss = epoch_time_loss / max(1, n_batch)
        train_loc_loss  = epoch_loc_loss / max(1, n_batch)

        val_loss, val_time_loss, val_loc_loss = eval_two_stage(
            model_time, model_loc, dl_valid, loss_cfg
        )

        evaluate_validation_ll = (
            epoch == 0
            or (epoch + 1) % ll_checkpoint_every == 0
            or epoch + 1 == max_epochs
        )
        if evaluate_validation_ll:
            val_ll_result = dataset_average_log_likelihood(
                model_time=model_time,
                model_loc=model_loc,
                data=validation_ll_data,
                dataset_name=ll_dataset_name,
                split_name="validation_checkpoint_subset",
                time_unit=ll_time_unit,
                device=device,
                event_batch_size=ll_checkpoint_event_batch_size,
                rtol=ll_checkpoint_rtol,
                atol=ll_checkpoint_atol,
                progress_every=0,
            )
            val_joint_ll = val_ll_result[
                "avg_loglik_joint_normalized"
            ]
            print(
                f"[Validation LL checkpoint] epoch {epoch:04d} | "
                f"joint={val_joint_ll:.6f}, "
                f"time={val_ll_result['avg_loglik_time']:.6f}, "
                "location(normalized)="
                f"{val_ll_result['avg_loglik_location_normalized']:.6f}, "
                f"sequences={len(validation_ll_data)}"
            )
            if val_joint_ll > best_val_joint_ll:
                best_val_joint_ll = val_joint_ll
                best_val_ll_result = val_ll_result
                best_ll_epoch = epoch
                best_time_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model_time.state_dict().items()
                }
                best_loc_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model_loc.state_dict().items()
                }

        print(
            f"epoch {epoch:04d} | "
            f"train_loss {train_loss:.4f} (time {train_time_loss:.4f}, loc {train_loc_loss:.4f}) | "
            f"val_loss {val_loss:.4f} "
            f"(time {val_time_loss:.4f}, loc {val_loc_loss:.4f})"
        )

    if best_time_state is None or best_loc_state is None:
        raise RuntimeError("No validation-LL checkpoint was recorded.")
    model_time.load_state_dict(best_time_state)
    model_loc.load_state_dict(best_loc_state)
    print(
        "[TwoStage FM] Restored best validation-LL checkpoint: "
        f"joint={best_val_joint_ll:.6f} at epoch {best_ll_epoch}, "
        f"time={best_val_ll_result['avg_loglik_time']:.6f}, "
        "location(normalized)="
        f"{best_val_ll_result['avg_loglik_location_normalized']:.6f}"
    )

    test_loss, test_time_loss, test_loc_loss = eval_two_stage(
        model_time, model_loc, dl_test, loss_cfg
    )
    print(
        f"[TwoStage FM] Test flow loss after checkpoint restore = "
        f"{test_loss:.4f} (time {test_time_loss:.4f}, "
        f"loc {test_loc_loss:.4f})"
    )

    time_rmse, loc_euclid = get_two_stage_prediction_error_sampled(
        model_time,
        model_loc,
        dl_test,
        num_steps_time=10,
        num_steps_loc=10,
        num_run = num_run,
    )
    print(
        f"[TwoStage Sampling Eval] time RMSE = {time_rmse:.6f}, "
        f"loc mean Euclidean distance = {loc_euclid:.6f}"
    )

    with open("res_st.txt", "a") as f:
        f.write(
            "[TwoStage FM] Restored best validation-LL checkpoint: "
            f"joint={best_val_joint_ll} at epoch {best_ll_epoch}, "
            f"time={best_val_ll_result['avg_loglik_time']}, "
            "location(normalized)="
            f"{best_val_ll_result['avg_loglik_location_normalized']}\n"
        )
        f.write(
            "[TwoStage FM] Test flow loss after checkpoint restore = "
            f"{test_loss} (time {test_time_loss}, loc {test_loc_loss})\n"
        )
        f.write(
            f"[TwoStage Sampling Eval] time RMSE = {time_rmse}, "
            f"loc mean Euclidean distance = {loc_euclid}\n"
        )

    checkpoint_info = {
        "criterion": "maximum validation avg_loglik_joint_normalized",
        "epoch": best_ll_epoch,
        "avg_loglik_time": best_val_ll_result["avg_loglik_time"],
        "avg_loglik_location_normalized": best_val_ll_result[
            "avg_loglik_location_normalized"
        ],
        "avg_loglik_joint_normalized": best_val_joint_ll,
        "validation_sequences_used": len(validation_ll_data),
        "validation_sequences_total": len(val_data),
        "evaluation_every_epochs": ll_checkpoint_every,
        "event_batch_size": ll_checkpoint_event_batch_size,
        "rtol": ll_checkpoint_rtol,
        "atol": ll_checkpoint_atol,
    }
    return time_rmse, loc_euclid, checkpoint_info

def test_event_st_two_stage(
        device,
        dataset,
        num_run=1,
        ll_event_batch_size=64,
        ll_rtol=1.0e-4,
        ll_atol=1.0e-5,
):
    data = load_earthquake_st_ori(str(DATA_ROOT / dataset))
    train_data = data["train"]
    val_data   = data["dev"]
    test_data  = data["test"]
    loc_dim = data["loc_dim"]
    max_len = data["max_len"] + 1
    loc_min = data["loc_min"]
    loc_max = data["loc_max"]

    model_time = SpatioTemporalTraceformer(
        loc_dim=loc_dim,
        d_model=64,
        n_heads=1,
        n_layers=3,
        d_ff=64,
        max_len=max_len,
        dropout=0.05,
        loc_min=loc_min,
        loc_max=loc_max,
    ).to(device)

    model_loc = SpatioTemporalTraceformer(
        loc_dim=loc_dim,
        d_model=64,
        n_heads=1,
        n_layers=3,
        d_ff=64,
        max_len=max_len,
        dropout=0.05,
        loc_min=loc_min,
        loc_max=loc_max,
    ).to(device)
    # Removed debugging marker in release.
    for m in (model_time, model_loc):
        m.set_log_transform(True)
        m.set_RN_train(False)
        m.set_AR_train(True)
        m.set_rescale(1.0)
        m.set_selection_combintation_weights(w_time_err=1.0, w_loc_err=1.0)

    preset = LL_DATASET_PRESETS[dataset.lower()]
    time_error, loc_error, checkpoint_info = run_two_stage(
        model_time,
        model_loc,
        train_data,
        val_data,
        test_data,
        max_len=max_len,
        learning_rate=1e-3,
        max_epochs=200,
        batch_size_train=32,
        batch_size_eval=256,
        num_run=num_run,
        ll_dataset_name=preset["display_name"],
        ll_time_unit=preset["time_unit"],
        ll_checkpoint_every=25,
        ll_checkpoint_max_sequences=32,
        ll_checkpoint_event_batch_size=ll_event_batch_size,
    )

    likelihood_result = dataset_average_log_likelihood(
        model_time=model_time,
        model_loc=model_loc,
        data=test_data,
        dataset_name=preset["display_name"],
        split_name="test",
        time_unit=preset["time_unit"],
        device=torch.device(device),
        event_batch_size=ll_event_batch_size,
        rtol=ll_rtol,
        atol=ll_atol,
        progress_every=10,
    )
    likelihood_result["checkpoint_selection"] = checkpoint_info
    print(json.dumps(likelihood_result, indent=2, ensure_ascii=False))
    likelihood_path = f"{dataset.lower()}_traceformer_likelihood_test.json"
    with open(likelihood_path, "w", encoding="utf-8") as handle:
        json.dump(likelihood_result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved likelihood result: {likelihood_path}")

    return time_error, loc_error, likelihood_result

#seeds_list = [42,39,41,16,46]
seeds_list = [16,42,39] #COVID
seeds_list = [46,41,39]

