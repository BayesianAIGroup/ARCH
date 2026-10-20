
from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# A) Helpers matching STSCPDataset
# ---------------------------

def gaussian2d_pdf_points(points, center, cov):
    inv_cov = np.linalg.inv(cov)
    sqrt_inv_det = 1.0 / np.sqrt(np.linalg.det(cov))
    delta = points - center[None, :]
    expo = -0.5 * np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    return (1.0 / (2.0 * np.pi)) * sqrt_inv_det * np.exp(expo)


def build_spatial_grid(x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), x_num=51, y_num=51):
    xs = np.linspace(x_lim[0], x_lim[1], x_num)
    ys = np.linspace(y_lim[0], y_lim[1], y_num)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    s_grids = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=-1)
    return xs, ys, XX, YY, s_grids


def stoi_from_grid(s, x_num, y_num):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    ij = np.round(s * np.array([[x_num - 1, y_num - 1]])).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, x_num - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, y_num - 1)
    return (ij[:, 0] + ij[:, 1] * x_num).astype(int)


def precompute_stscp_kernels(
    g0_cov,
    g2_cov,
    mu,
    gamma,
    x_num=51,
    y_num=51,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
):
    xs, ys, XX, YY, s_grids = build_spatial_grid(x_lim, y_lim, x_num, y_num)
    num_point = x_num * y_num

    g0_raw = gaussian2d_pdf_points(s_grids, np.array([0.5, 0.5]), g0_cov)
    g0_mat = g0_raw / np.sum(g0_raw) * x_num * y_num * mu

    g2_mats = np.zeros((num_point, num_point), dtype=np.float64)
    center_ref = np.array([0.5, 0.5], dtype=np.float64)

    for i, s in enumerate(s_grids):
        local_cov = g2_cov / (gamma * np.sum((s - center_ref) ** 2) + 1.0)
        g2_raw = gaussian2d_pdf_points(s_grids, s, local_cov)
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
        "x_lim": x_lim,
        "y_lim": y_lim,
    }


# ---------------------------
# B) STSCP intensity
# ---------------------------

def lambda_vector_stscp(t, events, *, beta, alpha, mu_vec, g0_mat, g2_mats, x_num, y_num, lamb_max=10.0):
    """
    Full grid intensity vector, matching STSCPDataset.lamb_St
    """
    if events is None or len(events) == 0:
        lam = mu_vec * np.exp(g0_mat * beta * t)
        return np.minimum(lamb_max, lam)

    hist = events[events[:, 0] < t]
    if hist.shape[0] == 0:
        lam = mu_vec * np.exp(g0_mat * beta * t)
        return np.minimum(lamb_max, lam)

    idx = stoi_from_grid(hist[:, 1:3], x_num, y_num)
    suppress = np.sum(g2_mats[idx], axis=0)   # [G]
    lam = mu_vec * np.exp(g0_mat * beta * t - alpha * suppress)
    lam = np.minimum(lamb_max, lam)
    return lam


def lambda_t_given_history(t_grid, events, *, beta, alpha, mu, gamma, g0_cov, g2_cov,
                           x_num=51, y_num=51, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), lamb_max=10.0):
    """
    λ(t|H), matching STSCPDataset.lamb_t
    """
    cache = precompute_stscp_kernels(
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        mu=mu,
        gamma=gamma,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
    )
    mu_vec = mu * np.ones(x_num * y_num, dtype=np.float64)

    t_grid = np.asarray(t_grid, dtype=np.float64)
    lam_t = np.empty_like(t_grid, dtype=np.float64)

    for j, t in enumerate(t_grid):
        lam_vec = lambda_vector_stscp(
            t,
            events,
            beta=beta,
            alpha=alpha,
            mu_vec=mu_vec,
            g0_mat=cache["g0_mat"],
            g2_mats=cache["g2_mats"],
            x_num=x_num,
            y_num=y_num,
            lamb_max=lamb_max,
        )
        lam_t[j] = np.sum(lam_vec) / x_num / y_num

    return lam_t


def lambda_txy_snapshot(x_grid, y_grid, t_snap, events, *, beta, alpha, mu, gamma, g0_cov, g2_cov,
                        x_num=51, y_num=51, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), lamb_max=10.0):
    """
    λ(t,x,y|H) snapshot via nearest discretized cell
    """
    cache = precompute_stscp_kernels(
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        mu=mu,
        gamma=gamma,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
    )
    mu_vec = mu * np.ones(x_num * y_num, dtype=np.float64)

    lam_vec = lambda_vector_stscp(
        t_snap,
        events,
        beta=beta,
        alpha=alpha,
        mu_vec=mu_vec,
        g0_mat=cache["g0_mat"],
        g2_mats=cache["g2_mats"],
        x_num=x_num,
        y_num=y_num,
        lamb_max=lamb_max,
    )

    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    idx = stoi_from_grid(pts, x_num, y_num)
    lam_xy = lam_vec[idx].reshape(len(y_grid), len(x_grid))
    return lam_xy


# ---------------------------
# C) Stable local-window thinning simulator
# ---------------------------

def simulate_stscp(
    t_max=30.0,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
    *,
    g0_cov=np.array([[0.02, 0.0], [0.0, 0.02]], dtype=np.float64),
    g2_cov=np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float64),
    alpha=0.5,
    beta=1.0,
    mu=0.2,
    gamma=1.0,
    lamb_max=10.0,
    x_num=51,
    y_num=51,
    seed=None,
    min_events=15,
    max_tries=20,
    dt_bound=0.5,
    verbose=False,
):
    """
    Stable simulator for STSCP using local-window thinning.
    """
    rng = np.random.default_rng(seed)

    cache = precompute_stscp_kernels(
        g0_cov=g0_cov,
        g2_cov=g2_cov,
        mu=mu,
        gamma=gamma,
        x_num=x_num,
        y_num=y_num,
        x_lim=x_lim,
        y_lim=y_lim,
    )
    s_grids = cache["s_grids"]
    g0_mat = cache["g0_mat"]
    g2_mats = cache["g2_mats"]
    g0_max = np.max(g0_mat)

    mu_vec = mu * np.ones(x_num * y_num, dtype=np.float64)

    for _ in range(max_tries):
        t = 0.0
        events = []

        while t < t_max:
            hist_arr = np.array(events, dtype=np.float64) if len(events) > 0 else np.zeros((0, 3), dtype=np.float64)

            # current full intensity
            lam_vec_now = lambda_vector_stscp(
                t,
                hist_arr,
                beta=beta,
                alpha=alpha,
                mu_vec=mu_vec,
                g0_mat=g0_mat,
                g2_mats=g2_mats,
                x_num=x_num,
                y_num=y_num,
                lamb_max=lamb_max,
            )
            lam_t_now = np.sum(lam_vec_now) / x_num / y_num

            if lam_t_now <= 1e-12:
                break

            # local future upper bound on [t, t + dt_bound]
            # because only the exp(g0_i * beta * t) term can grow with time,
            # while suppression term does not decrease before the next accepted event.
            growth = np.exp(g0_max * beta * dt_bound)
            M = lam_t_now * growth * 1.05

            # propose next candidate within this local window
            delta_t = rng.exponential(scale=1.0 / M)

            if delta_t > dt_bound:
                t += dt_bound
                continue

            t_candidate = t + delta_t
            if t_candidate > t_max:
                break

            lam_vec_cand = lambda_vector_stscp(
                t_candidate,
                hist_arr,
                beta=beta,
                alpha=alpha,
                mu_vec=mu_vec,
                g0_mat=g0_mat,
                g2_mats=g2_mats,
                x_num=x_num,
                y_num=y_num,
                lamb_max=lamb_max,
            )
            lam_t_cand = np.sum(lam_vec_cand) / x_num / y_num

            if rng.uniform() < lam_t_cand / M:
                p = lam_vec_cand / np.sum(lam_vec_cand)
                i = np.argmax(rng.multinomial(1, p))
                x, y = s_grids[i]
                events.append((t_candidate, x, y))

                if verbose:
                    print(f"accepted: t={t_candidate:.4f}, lambda_t={lam_t_cand:.4f}, n={len(events)}")

            t = t_candidate

        events = np.array(events, dtype=np.float32)
        if events.shape[0] >= min_events:
            return events, True

    return np.zeros((0, 3), dtype=np.float32), False


def generate_split(n_sequences, *, base_seed, **sim_kwargs):
    res = []
    n_fail = 0
    for i in range(n_sequences):
        events, success = simulate_stscp(seed=base_seed + i, **sim_kwargs)
        if success:
            print(len(events))
            res.append(events)
        else:
            n_fail += 1

        if i % 50 == 0:
            print(f"{i}/{n_sequences} | success={len(res)} | fail={n_fail}")
    return res


# ---------------------------
# D) Main: PKLs + plots
# ---------------------------

def main(out_dir="./stscp_out", make_plots=True):
    os.makedirs(out_dir, exist_ok=True)

    # sim_kwargs = dict(
    #     t_max=50.0,
    #     x_lim=(0.0, 1.0),
    #     y_lim=(0.0, 1.0),
    #     g0_cov=np.array([[1, 0.0], [0.0, 1]], dtype=np.float64),
    #     g2_cov=np.array([[0.85, 0.0], [0.0, 0.85]], dtype=np.float64),
    #     alpha=0.2,
    #     beta=0.2,
    #     mu=1.0,
    #     gamma=2.0,
    #     lamb_max=10.0,
    #     x_num=51,
    #     y_num=51,
    #     min_events=15,
    #     dt_bound=0.5,
    # )

    sim_kwargs = dict(
        t_max=100.0,
        x_lim=(0.0, 1.0),
        y_lim=(0.0, 1.0),
        g0_cov=np.array([[0.25, 0.0], [0.0, 0.25]], dtype=np.float64),
        g2_cov=np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64),
        alpha=0.4,
        beta=0.2,
        mu=1.0,
        gamma=2.0,
        lamb_max=10.0,
        x_num=51,
        y_num=51,
        min_events=15,
        dt_bound=0.5,
    )

    train = generate_split(10000, base_seed=12345, **sim_kwargs)
    val   = generate_split(200,  base_seed=22345, **sim_kwargs)
    test  = generate_split(200,  base_seed=32345, **sim_kwargs)

    with open(os.path.join(out_dir, "data_train.pkl"), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "data_val.pkl"), "wb") as f:
        pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "data_test.pkl"), "wb") as f:
        pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Generated: train={len(train)}, val={len(val)}, test={len(test)}")

    if not make_plots or len(test) == 0:
        return

    example = test[min(15, len(test) - 1)]
    T = sim_kwargs["t_max"]

    t_grid = np.linspace(0, T, 600)
    lam_t = lambda_t_given_history(
        t_grid,
        example,
        beta=sim_kwargs["beta"],
        alpha=sim_kwargs["alpha"],
        mu=sim_kwargs["mu"],
        gamma=sim_kwargs["gamma"],
        g0_cov=sim_kwargs["g0_cov"],
        g2_cov=sim_kwargs["g2_cov"],
        x_num=sim_kwargs["x_num"],
        y_num=sim_kwargs["y_num"],
        x_lim=sim_kwargs["x_lim"],
        y_lim=sim_kwargs["y_lim"],
        lamb_max=sim_kwargs["lamb_max"],
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t_grid, lam_t)
    for te in example[:, 0]:
        plt.axvline(te, linewidth=0.5, alpha=0.25)
    plt.xlabel("t")
    plt.ylabel("lambda(t | H)")
    plt.title("STSCP temporal intensity λ(t|H)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_t_given_H.png"), dpi=180)
    plt.close()

    xg = np.linspace(sim_kwargs["x_lim"][0], sim_kwargs["x_lim"][1], 140)
    yg = np.linspace(sim_kwargs["y_lim"][0], sim_kwargs["y_lim"][1], 140)

    if example.shape[0] >= 3:
        t1 = float(min(example[0, 0] + 0.2, 0.3 * T))
        t2 = float(min(example[len(example) // 2, 0] + 0.2, 0.6 * T))
        t3 = float(min(example[-1, 0] + 0.2, 0.9 * T))
    else:
        t1, t2, t3 = 0.25 * T, 0.5 * T, 0.75 * T

    for k, ts in enumerate([t1, t2, t3], start=1):
        lam_xy = lambda_txy_snapshot(
            xg, yg, ts, example,
            beta=sim_kwargs["beta"],
            alpha=sim_kwargs["alpha"],
            mu=sim_kwargs["mu"],
            gamma=sim_kwargs["gamma"],
            g0_cov=sim_kwargs["g0_cov"],
            g2_cov=sim_kwargs["g2_cov"],
            x_num=sim_kwargs["x_num"],
            y_num=sim_kwargs["y_num"],
            x_lim=sim_kwargs["x_lim"],
            y_lim=sim_kwargs["y_lim"],
            lamb_max=sim_kwargs["lamb_max"],
        )

        plt.figure(figsize=(5.2, 4.5))
        plt.imshow(
            lam_xy,
            origin="lower",
            extent=(sim_kwargs["x_lim"][0], sim_kwargs["x_lim"][1],
                    sim_kwargs["y_lim"][0], sim_kwargs["y_lim"][1]),
            aspect="auto",
        )
        plt.colorbar(label="lambda(t, x, y | H)")
        plt.scatter(example[example[:, 0] < ts, 1], example[example[:, 0] < ts, 2], s=6, alpha=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"STSCP λ(t,x,y|H) at t={ts:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"lambda_txy_snapshot_{k}.png"), dpi=180)
        plt.close()


if __name__ == "__main__":
    main()