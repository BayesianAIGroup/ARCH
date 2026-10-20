
from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path
from datetime import datetime
from dateutil import parser
import numpy as np
import joblib as jbl
import os
from dataclasses import dataclass
import torch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#True: this token should be masked out
def make_causal_mask(T: int, device):
    # [T, T] with True where we should mask (future positions)
    # nn.Transformer expects float mask with -inf on blocked positions (or bool depending on API)
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()  # upper triangle is future
    return mask

#True: this token should be masked out
def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # lengths: [B], returns [B, max_len] where True = pad
    idxs = torch.arange(max_len, device=lengths.device)[None, :] #[1, max_len]
    return idxs >= lengths[:, None] #[B, 1]


def _parse_ts(s):
    # try ISO with optional Z
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        pass
    # try common formats
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # fallback to dateutil if available
    try:
        return parser.parse(s)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {s}")


#../ExtractEvents/BPIC-2017-action-trace.pkl
def get_my_data(pickle_file, data_file_out):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    token_dict = data['token_dict'] #{token-name: id}
    trace_list = data['trace_list'] #list of (token-id, time) list

    token_id_name = {}
    for name,id in token_dict.items():
        token_id_name[id] = name

    seqs_token_lag = [] #(token-id, lag) seq, lag is computed by seconds
    for each_seq in trace_list:
        first_token = each_seq[0][0]
        seq = []
        if first_token == 0:
            time_points = [_parse_ts(each_item[1]) for each_item in each_seq]
            for j in range(1, len(time_points)):
                lag = time_points[j] - time_points[j-1]
                seq.append((each_seq[j][0], lag.total_seconds()))
            seqs_token_lag.append(seq)
        else:
            print('not starting from Application Creation')

    data = {'token_name2id': token_dict, 'token_id2name': token_id_name, 'trace_list': seqs_token_lag}
    pkl_path = data_file_out if data_file_out.endswith(".pkl") else data_file_out + ".pkl"
    with open(pkl_path, "wb") as pf:
        pickle.dump(data, pf, protocol=pickle.HIGHEST_PROTOCOL)
    print('%d actions (event types)'%len(token_id_name))
    print('%d traces'%len(seqs_token_lag))


# ----------------------------
# Losses
# ----------------------------
@dataclass
class LossConfig:
    token_label_smoothing: float = 0.0
    time_loss_weight: float = 1.0
    token_loss_weight: float = 1.0
    pad_id: int = 0


def lognormal_nll(target_dt: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of LogNormal(target_dt; mu, sigma), where mu/log_sigma are
    params of the *log-space* normal: log(dt) ~ N(mu, sigma^2).
    """
    eps = 1e-10
    z = (torch.log(target_dt + eps) - mu) / (log_sigma.exp() + 1e-8)
    nll = 0.5 * z**2 + log_sigma + torch.log(target_dt + eps)
    return nll  # elementwise

def normal_nll(target_dt: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    '''
    Negaitve log-likelihood of Normal distribution, where log-sigma is a dummpy/useless
    '''
    z = target_dt - mu
    nll = 0.5*z**2 #MSE is performing better
    return nll

def token_ce_with_smoothing(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, smoothing: float = 0.0):
    """
    Cross-entropy with optional label smoothing. Computes per-position loss.
    logits: [*, V], targets: [*]
    targets: [*]
    ignore_index: ignored token ID
    Returns: [*] loss values
    """
    V = logits.size(-1) #number of tokens/classes
    if smoothing <= 0.0:
        #when target's token == ignore_index, ignore the computation of the loss (negative ll) for this token
        #logits: (..., V)
        #targets: (...)
        loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)
        return loss

    # Manual label smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(logits) #[N, V]
        true_dist.fill_(smoothing / (V - 1))
        mask = targets.ne(ignore_index) #(N,)
        true_dist[mask, targets[mask]] = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1)
    # force ignore_index positions to 0
    loss = torch.where(targets.eq(ignore_index), torch.zeros_like(loss), loss)
    return loss


def load_earthquake_st_ori(
    data_folder="./earthquake",
    print_info=True,
):
    """
    load data_train/val/test.pkl
    output:
        data = {
            'train': list of seq,
            'dev':   list of seq,
            'test':  list of seq,
        }
    """

    def load_split(name):
        path = os.path.join(data_folder, f"data_{name}.pkl")
        with open(path, "rb") as f:
            seqs = pickle.load(f)
        seqs = [[list(e) for e in u] for u in seqs]
        return seqs

    train_raw = load_split("train")
    val_raw = load_split("val")
    test_raw = load_split("test")

    all_locs = []
    all_delta = []

    def to_delta_and_collect(seqs):
        out = []
        for u in seqs:
            if len(u) == 0:
                continue
            new_seq = []
            prev_t = None
            for e in u:
                t_abs = e[0]
                loc_vec = e[1:]  # [lng, lat]
                if prev_t is None:
                    delta = 1e-6
                else:
                    delta = t_abs - prev_t
                    if delta <= 0:
                        delta = 1e-6
                prev_t = t_abs
                all_locs.append(loc_vec)
                all_delta.append(delta)
                new_seq.append((delta, loc_vec))
            out.append(new_seq)
        return out

    # transfer to delta t
    train_seq = to_delta_and_collect(train_raw)
    val_seq = to_delta_and_collect(val_raw)
    test_seq = to_delta_and_collect(test_raw)

    all_locs = np.array(all_locs, dtype=np.float32)
    all_delta = np.array(all_delta, dtype=np.float32)

    loc_min = all_locs.min(axis=0)
    loc_max = all_locs.max(axis=0)

    if print_info:
        print(f"[Earthquake] loc_min = {loc_min}, loc_max = {loc_max}")
        print(f"[Earthquake] delta_min = {all_delta.min()}, delta_max = {all_delta.max()}")

    eps = 1e-8

    def normalize_and_pack(seq_list):
        data_processed = []
        max_len = 0
        loc_dim = len(loc_min)
        for seq in seq_list:
            if len(seq) == 0:
                continue
            deltas = []
            locs = []
            for (delta, loc_vec) in seq:
                deltas.append(delta)
                loc_vec = np.array(loc_vec, dtype=np.float32)
                loc_norm = (loc_vec - loc_min) / (loc_max - loc_min + eps)
                locs.append(loc_norm)
            deltas = torch.tensor(deltas, dtype=torch.float32)
            locs = torch.tensor(locs, dtype=torch.float32)  # [L, loc_dim]
            L = deltas.shape[0]
            max_len = max(max_len, L)
            data_processed.append(
                {
                    "length": L,
                    "deltas": deltas,
                    "locs": locs,
                }
            )
        return data_processed, max_len, loc_dim


    # range normlize lon and lat, not t
    train_data, max_len_tr, loc_dim_tr = normalize_and_pack(train_seq)
    val_data, max_len_val, loc_dim_val = normalize_and_pack(val_seq)
    test_data, max_len_te, loc_dim_te = normalize_and_pack(test_seq)

    loc_dim = loc_dim_tr

    max_len = max(max_len_tr, max_len_val, max_len_te)

    if print_info:
        def stat(split_name, data_list):
            lens = [d["length"] for d in data_list]
            print(
                f"{split_name}: num_seqs={len(lens)}, "
                f"min_len={np.min(lens)}, max_len={np.max(lens)}, mean_len={np.mean(lens):.1f}"
            )

        stat("train", train_data)
        stat("dev", val_data)
        stat("test", test_data)
        print(f"loc_dim = {loc_dim}, max_len = {max_len}")

    return {
        "train": train_data,
        "dev": val_data,
        "test": test_data,
        "loc_dim": loc_dim,
        "max_len": max_len,
        "loc_min": loc_min,
        "loc_max": loc_max,
    }


def load_earthquake_st(data_dir="./dataset/Earthquake"):
    with open(os.path.join(data_dir, "data_train.pkl"), "rb") as f:
        train_raw = pickle.load(f)
    with open(os.path.join(data_dir, "data_val.pkl"), "rb") as f:
        val_raw = pickle.load(f)
    with open(os.path.join(data_dir, "data_test.pkl"), "rb") as f:
        test_raw = pickle.load(f)

    # train_raw / val_raw / test_raw: list of sequences
    def add_delta(seq):
        """Convert a sequence of [absolute_time, longitude, latitude, ...] events with time at index 0 to time gaps."""
        new_seq = []
        prev_t = None
        for e in seq:
            t_abs = e[0]
            if prev_t is None:
                delta = t_abs
            else:
                delta = t_abs - prev_t
            prev_t = t_abs
            loc_vec = e[1:]
            new_seq.append([t_abs, delta, *loc_vec])
        return new_seq

    train_raw2 = [add_delta(u) for u in train_raw]
    val_raw2   = [add_delta(u) for u in val_raw]
    test_raw2  = [add_delta(u) for u in test_raw]

    data_all = train_raw2 + val_raw2 + test_raw2
    dim_total = len(data_all[0][0])  # Earthquake: 0:t_abs, 1:delta_t, 2:lng, 3:lat

    Max = []
    Min = []
    for m in range(dim_total):
        if m > 0:
            vals_m = [e[m] for seq in data_all for e in seq]
            Max.append(max(vals_m))
            Min.append(min(vals_m))
        else:
            Max.append(1.0)
            Min.append(0.0)

    Max = np.array(Max, dtype=np.float32)
    Min = np.array(Min, dtype=np.float32)

    assert Min[1] > 0, "delta_t must be strictly positive for log/likelihood calculations"

    def normalize_all(seq_list):
        out = []
        max_len = 0
        for u in seq_list:
            if len(u) == 0:
                continue
            u_norm = []
            for e in u:
                e = np.array(e, dtype=np.float32)
                e_norm = (e - Min) / (Max - Min + 1e-8)
                u_norm.append(e_norm)
            out.append(u_norm)
            max_len = max(max_len, len(u_norm))
        return out, max_len

    train_norm, max_len_tr = normalize_all(train_raw2)
    val_norm,   max_len_v  = normalize_all(val_raw2)
    test_norm,  max_len_te = normalize_all(test_raw2)

    max_len = max(max_len_tr, max_len_v, max_len_te)
    loc_dim = dim_total - 2  # Earthquake: 2

    def pack_for_traceformer(seq_norm_list):
        data_processed = []
        for u in seq_norm_list:
            # u: list of [t_abs_norm(=t_abs), delta_norm, loc_norm...]
            u = np.stack(u, axis=0)  # [L, dim_total]
            deltas = u[:, 1].astype(np.float32)
            locs   = u[:, 2:].astype(np.float32)
            L = len(deltas)
            data_processed.append(
                {
                    "length": L,
                    "deltas": torch.from_numpy(deltas),       # [L]
                    "locs":   torch.from_numpy(locs),         # [L, loc_dim]
                }
            )
        return data_processed

    train_data = pack_for_traceformer(train_norm)
    val_data   = pack_for_traceformer(val_norm)
    test_data  = pack_for_traceformer(test_norm)

    return train_data, val_data, test_data, max_len, loc_dim, Max, Min

def load_data_zhimeng( data_name, data_folder = './event-data', fold = 0, print_info = True):
    valid_data_names = ['retweet', 'so', 'mimic', 'BiVariate_Case0']
    if data_name not in valid_data_names:
        raise NameError('Invalid data name. Available data names: ', valid_data_names)

    data_path = os.path.join( data_folder, '%s_5_folds.jbl' % data_name )
    if not os.path.exists(data_path):
        data_path = os.path.join( data_folder, '%s.jbl' % data_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        data = jbl.load( data_path )
    else:
        data = jbl.load( data_path)
        data = data[fold]

    if print_info:
        print('loaded data %s, fold %d' % (data_name, fold))
        print('fold = %g' % fold)
        print('num Types = %d' % data['num_types'])
        totalNumEvents = 0
        totalNumSeqs = 0
        for sub in ['train', 'dev', 'test']:
            lengths = [ len( seq) for seq in data[sub]['arr_list_timestamps']]
            min_len = np.min( lengths)
            max_len = np.max( lengths)
            mean_len = np.mean( lengths)
            subTotalEvents = np.sum( lengths)
            totalNumEvents += subTotalEvents
            totalNumSeqs += len( lengths)
            print('data_name = %s, sub = %s, num_seqs = %d, num_events = %d, min_len = %d, max_len =%d, mean_len =%d' % (
                data_name,sub, len( lengths), subTotalEvents, min_len, max_len, mean_len))
        print( 'totalNumSeqs = %d, totalNumEvents= = %d' % ( totalNumSeqs, totalNumEvents))
    return data


def save_two_stage_checkpoint(model_time, model_loc, ckpt_path: str, meta: dict = None):
    """Save parameters of both models to ckpt_path."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "model_time_state_dict": model_time.state_dict(),
        "model_loc_state_dict": model_loc.state_dict(),
    }
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, ckpt_path)
    print(f"[Checkpoint] Saved two-stage models to: {ckpt_path}")


