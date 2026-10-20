
from release_paths import DATA_ROOT, CHECKPOINT_ROOT, data_path
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# A) Hawkes intensity (for visualization)
# ---------------------------

def lambda_txy_snapshot(x_grid, y_grid, t_snap, events, *, mu, alpha, beta, sigma):
    """Compute λ(t_snap, x, y | H) on a grid."""
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    lam = np.full_like(X, mu, dtype=np.float64)

    if events.shape[0] == 0:
        return lam

    ti = events[:, 0].astype(np.float64)
    xi = events[:, 1].astype(np.float64)
    yi = events[:, 2].astype(np.float64)

    mask = ti < t_snap
    if not np.any(mask):
        return lam

    ti = ti[mask]; xi = xi[mask]; yi = yi[mask]
    temp = alpha * np.exp(-beta * (t_snap - ti))  # [n]

    dx = X[..., None] - xi[None, None, :]
    dy = Y[..., None] - yi[None, None, :]
    dist_sq = dx * dx + dy * dy
    spat = (1.0 / (2.0 * np.pi * sigma * sigma)) * np.exp(-dist_sq / (2.0 * sigma * sigma))

    lam = lam + np.sum(spat * temp[None, None, :], axis=2)
    return lam


def _stdnorm_cdf(z):
    # Φ(z) via erf (no scipy dependency)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def _rect_mass_2d_gaussian(xi, yi, sigma, x_lim, y_lim):
    """Integral of N([xi,yi], sigma^2 I) density over the rectangle."""
    xmin, xmax = x_lim
    ymin, ymax = y_lim

    ax = (xmax - xi) / sigma
    bx = (xmin - xi) / sigma
    ay = (ymax - yi) / sigma
    by = (ymin - yi) / sigma

    mx = _stdnorm_cdf(ax) - _stdnorm_cdf(bx)
    my = _stdnorm_cdf(ay) - _stdnorm_cdf(by)
    return mx * my

def lambda_t_given_history(t_grid, events, *, mu, alpha, beta, sigma, x_lim, y_lim):
    """
    Temporal intensity λ(t|H) = ∬ λ(t,x,y|H) dx dy over spatial window.
    """
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    area = (xmax - xmin) * (ymax - ymin)

    if events.shape[0] == 0:
        return np.full_like(t_grid, mu * area, dtype=np.float64)

    ti = events[:, 0].astype(np.float64)
    xi = events[:, 1].astype(np.float64)
    yi = events[:, 2].astype(np.float64)

    mass = np.array(
        [_rect_mass_2d_gaussian(xi[k], yi[k], sigma, x_lim, y_lim) for k in range(len(ti))],
        dtype=np.float64,
    )

    lam_t = np.empty_like(t_grid, dtype=np.float64)
    for j, t in enumerate(t_grid):
        mask = ti < t
        if not np.any(mask):
            lam_t[j] = mu * area
        else:
            lam_t[j] = mu * area + np.sum(alpha * np.exp(-beta * (t - ti[mask])) * mass[mask])
    return lam_t

# ---------------------------
# B) Fast simulator: Hawkes Poisson-cluster (branching) representation
# ---------------------------

def simulate_hawkes_cluster(
    t_max=30.0,
    x_lim=(0.0, 1.0),
    y_lim=(0.0, 1.0),
    *,
    mu=1.0,          # background intensity density (per time per area)
    alpha=0.72,      # excitation amplitude in g(t)=alpha*exp(-beta t)
    beta=1.2,        # decay rate
    sigma=0.05,      # spatial std
    seed=None,
    min_events=15,
    max_tries=20,
):
    """
    Simulate ST Hawkes on a rectangle using Poisson cluster representation.
    Output: [N,3] sorted by time, columns [t,x,y].
    """
    rng = np.random.default_rng(seed)
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    area = (xmax - xmin) * (ymax - ymin)

    branching_mean = alpha / beta  # stability requires < 1

    for _ in range(max_tries):
        n0 = rng.poisson(mu * area * t_max)
        if n0 == 0:
            continue

        t0 = rng.uniform(0.0, t_max, size=n0)
        x0 = rng.uniform(xmin, xmax, size=n0)
        y0 = rng.uniform(ymin, ymax, size=n0)

        t_list = t0.tolist()
        x_list = x0.tolist()
        y_list = y0.tolist()

        queue = list(range(n0))

        while queue:
            idx = queue.pop()
            ti, xi, yi = t_list[idx], x_list[idx], y_list[idx]

            k = rng.poisson(branching_mean)
            if k == 0:
                continue

            dt = rng.exponential(1.0 / beta, size=k)
            tcand = ti + dt
            tcand = tcand[tcand < t_max]
            if tcand.size == 0:
                continue

            for tj in tcand:
                for _attempt in range(30):
                    xj = xi + rng.normal(0.0, sigma)
                    yj = yi + rng.normal(0.0, sigma)
                    if (xmin <= xj <= xmax) and (ymin <= yj <= ymax):
                        t_list.append(float(tj))
                        x_list.append(float(xj))
                        y_list.append(float(yj))
                        queue.append(len(t_list) - 1)
                        break

        events = np.stack([np.array(t_list), np.array(x_list), np.array(y_list)], axis=1)
        events = events[np.argsort(events[:, 0])].astype(np.float32)

        if events.shape[0] >= min_events:
            return events, True

    print('failed')
    return np.zeros((0, 3), dtype=np.float32), False

def generate_split(n_sequences, *, base_seed, **sim_kwargs):
    res = []
    for i in range(n_sequences):
        events, success = simulate_hawkes_cluster(seed=base_seed + i, **sim_kwargs)
        if success:
            res.append(events)
    return res
# ---------------------------
# C) (Optional) Ogata thinning version (closer to your doc, but slower)
# ---------------------------

def hawkes_intensity_thinning(t, x, y, history, *, mu, alpha, beta, sigma):
    lam = mu
    if len(history) == 0:
        return lam
    for (ti, xi, yi) in history:
        if ti >= t:
            continue
        dt = t - ti
        temp = math.exp(-beta * dt)
        dist_sq = (x - xi) ** 2 + (y - yi) ** 2
        spat = (1.0 / (2.0 * math.pi * sigma * sigma)) * math.exp(-dist_sq / (2.0 * sigma * sigma))
        lam += alpha * temp * spat
    return lam

def hawkes_upper_bound_thinning(t, history, *, mu, alpha, beta, sigma, safety=1.2):
    spatial_peak = 1.0 / (2.0 * math.pi * sigma * sigma)
    s = 0.0
    for (ti, _, _) in history:
        if ti >= t:
            continue
        s += alpha * math.exp(-beta * (t - ti)) * spatial_peak
    return (mu + s) * safety

def simulate_hawkes_thinning(t_max, x_lim, y_lim, *, mu, alpha, beta, sigma, seed=None):
    rng = np.random.default_rng(seed)
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    area = (xmax - xmin) * (ymax - ymin)

    t = 0.0
    events = []

    while t < t_max:
        L = hawkes_upper_bound_thinning(t, events, mu=mu, alpha=alpha, beta=beta, sigma=sigma, safety=1.2)
        rate_max = L * area
        if rate_max <= 1e-12:
            break

        t += rng.exponential(1.0 / rate_max)
        if t > t_max:
            break

        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)

        true_lam = hawkes_intensity_thinning(t, x, y, events, mu=mu, alpha=alpha, beta=beta, sigma=sigma)
        p = min(1.0, true_lam / L)
        if rng.random() < p:
            events.append((t, x, y))

    return np.array(events, dtype=np.float32)

# ---------------------------
# D) Main: generate PKLs + plots
# ---------------------------

def main(out_dir="./", make_plots=True):
    os.makedirs(out_dir, exist_ok=True)

    # Parameters (stable: alpha/beta = 0.6 < 1)
    sim_kwargs = dict(
        t_max=60.0,
        x_lim=(0.0, 1.0),
        y_lim=(0.0, 1.0),
        mu=1.0,
        alpha=0.72,
        beta=1.2,
        sigma=0.05,
        min_events=15,
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

    if not make_plots:
        return

    example = test[15]
    T = sim_kwargs["t_max"]

    # λ(t|H)
    t_grid = np.linspace(0, T, 600)
    lam_t = lambda_t_given_history(
        t_grid,
        example,
        mu=sim_kwargs["mu"],
        alpha=sim_kwargs["alpha"],
        beta=sim_kwargs["beta"],
        sigma=sim_kwargs["sigma"],
        x_lim=sim_kwargs["x_lim"],
        y_lim=sim_kwargs["y_lim"],
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t_grid, lam_t)
    for te in example[:, 0]:
        plt.axvline(te, linewidth=0.5, alpha=0.25)
    plt.xlabel("t")
    plt.ylabel("lambda(t | H)  (integrated over space)")
    plt.title("Temporal intensity λ(t|H) with event times")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_t_given_H.png"), dpi=180)
    plt.close()

    # λ(t,x,y|H) heatmaps
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
            mu=sim_kwargs["mu"],
            alpha=sim_kwargs["alpha"],
            beta=sim_kwargs["beta"],
            sigma=sim_kwargs["sigma"],
        )
        plt.figure(figsize=(5.2, 4.5))
        plt.imshow(
            lam_xy,
            origin="lower",
            extent=(sim_kwargs["x_lim"][0], sim_kwargs["x_lim"][1], sim_kwargs["y_lim"][0], sim_kwargs["y_lim"][1]),
            aspect="auto",
        )
        plt.colorbar(label="lambda(t, x, y | H)")
        plt.scatter(example[example[:, 0] < ts, 1], example[example[:, 0] < ts, 2], s=6, alpha=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Spatial intensity λ(t,x,y|H) at t={ts:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"lambda_txy_snapshot_{k}.png"), dpi=180)
        plt.close()


if __name__ == "__main__":
    main(out_dir="./stpp_out", make_plots=True)