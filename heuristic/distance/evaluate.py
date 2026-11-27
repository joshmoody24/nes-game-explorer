"""
Held-out tests of model.npz: (1) does embedding distance track steps apart, (2) do embedding clusters match rooms,
(3) do learned-novelty scores point at stepping stones.
Usage: JAX_PLATFORMS=cpu python heuristic/distance/evaluate.py [distance] [room] [stone]
"""
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from scipy.stats import pearsonr, spearmanr
from common import runs, load, room
from train import encode, init, features, sample_pairs, trajectory_bounds, stack
from features_auc import auc

model = np.load(Path(__file__).parent / "model.npz")
chosen, mode = model["chosen"], jnp.asarray(model["mode"])
params = [(jnp.asarray(model[f"w{n}"]), jnp.asarray(model[f"b{n}"])) for n in range(sum(k.startswith("w") for k in model))]
embed_batch = jax.jit(lambda params, raw: encode(params, raw, mode))
untrained = init(jax.random.PRNGKey(0), params[0][0].shape[0])


def embed(memory, params=params):
    return np.concatenate([np.asarray(embed_batch(params, jnp.asarray(memory[k:k + 4096, chosen]))) for k in range(0, len(memory), 4096)])


def raw_pca(memory, dimensions=32):
    """Baseline: the model's input features, projected to as many dimensions as the embedding."""
    x = np.asarray(features(jnp.asarray(memory[:, chosen]), mode))
    x = x - x.mean(0)
    return x @ np.linalg.svd(x, full_matrices=False)[2][:dimensions].T


def squared_distances(x, centers):
    return (x ** 2).sum(1)[:, None] - 2 * x @ centers.T + (centers ** 2).sum(1)[None]


def kmeans(x, k, rng, iterations=30):
    centers = x[[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.maximum(squared_distances(x, centers).min(1), 0)
        centers = np.vstack([centers, x[rng.choice(len(x), p=d / d.sum())]])
    for _ in range(iterations):
        label = squared_distances(x, centers).argmin(1)
        centers = np.array([x[label == c].mean(0) if (label == c).any() else centers[c] for c in range(k)])
    return label


def leader(x, radius):
    """A new place whenever a state is farther than radius from every existing place's first state."""
    centers, label = [], np.empty(len(x), int)
    for n, v in enumerate(x):
        d = np.linalg.norm(np.array(centers) - v, axis=1) if centers else np.array([np.inf])
        if d.min() > radius:
            centers.append(v); label[n] = len(centers) - 1
        else:
            label[n] = d.argmin()
    return label


def purity(cluster, truth):
    return sum(np.bincount(np.unique(truth, return_inverse=True)[1][cluster == c]).max() for c in np.unique(cluster)) / len(truth)


def distance_test(test):
    memory, trajectory, step = stack(test)
    z = embed(memory)
    starts, ends = trajectory_bounds(trajectory)
    i, j, target = sample_pairs(np.random.default_rng(1), step, starts, ends, 20000)
    predicted = np.linalg.norm(z[i] - z[j], axis=1)
    print(f"distance: {len(memory)} held-out states; predicted vs log(1+steps apart): pearson {pearsonr(predicted, target)[0]:.3f}, "
          f"spearman {spearmanr(predicted, target)[0]:.3f}; vs raw steps spearman {spearmanr(predicted, np.expm1(target))[0]:.3f}")
    for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 200), (200, 10000)]:
        steps = np.expm1(target)
        m = (steps >= lo) & (steps < hi)
        print(f"  {lo}-{hi} steps apart: n={m.sum()}, mean predicted {predicted[m].mean():.2f} (true log mean {target[m].mean():.2f})")


def room_test(test):
    rng = np.random.default_rng(2)
    rows = {}
    for path in test:
        memory = load(path)["memory"]
        z, truth = embed(memory), room(memory)
        rooms = len(np.unique(truth))
        for per_room in [2, 5, 10, 15]:
            rows.setdefault(f"k-means, {per_room} per room", []).append((purity(kmeans(z, per_room * rooms, rng), truth), per_room))
        for per_room in [5, 10]:
            for name, baseline in [("untrained encoder", embed(memory, untrained)), ("raw input features, PCA 32", raw_pca(memory))]:
                rows.setdefault(f"baseline {name}: k-means, {per_room} per room", []).append((purity(kmeans(baseline, per_room * rooms, rng), truth), per_room))
        for radius in [2.0, 3.0, 4.0, 5.0]:
            label = leader(z, radius)
            rows.setdefault(f"leader radius {radius}", []).append((purity(label, truth), len(np.unique(label)) / rooms))
    print("rooms (held-out runs, trajectory states; purity vs (area,y,x)):")
    for name, values in rows.items():
        values = np.array(values)
        print(f"  {name}: purity {values[:, 0].mean():.1%} (min {values[:, 0].min():.1%}), places per room {values[:, 1].mean():.1f}")


def stone_test(test):
    rows = {}
    for path in test:
        d = load(path)
        first = int(d["first"])
        if first < 0:
            continue
        chain, i = set(), d["parents"][first]
        while i >= 0 and i not in chain:
            chain.add(i); i = d["parents"][i]
        z = embed(d["save_memory"][:first + 1])
        z_start = embed(d["start_memory"][None])[0]
        pairwise = np.linalg.norm(z[:, None] - z[None], axis=-1)
        earlier = np.where(np.tri(len(z), k=-1, dtype=bool), pairwise, np.inf)
        from_start = np.linalg.norm(z - z_start, axis=1)
        farthest_before = np.r_[0, np.maximum.accumulate(from_start)[:-1]]
        chosen_count = d["times_chosen"][:first + 1].astype(float)
        scores = {
            "embedding distance from start": from_start,
            "new record distance from start at creation": from_start - farthest_before,
            "distance-from-start rank among saves existing at creation": np.array([(from_start[:n + 1] <= from_start[n]).mean() for n in range(len(z))]),
            "creation order (reference)": np.arange(len(z), dtype=float),
            "distance to nearest earlier save (frontier)": np.r_[0, earlier[1:].min(1)],
            "mean distance to all other saves (isolation)": pairwise.mean(1),
            "times chosen": chosen_count,
            "classic weight 1/sqrt(1+times chosen)": 1 / np.sqrt(1 + chosen_count),
            "oracle, not available in a run: closeness to the first success": -np.linalg.norm(z - z[first], axis=1),
            "path length (reference)": d["path_length"][:first + 1].astype(float),
        }
        label = np.array([i in chain for i in range(first)])
        for name, score in scores.items():
            stone_auc = auc(score[:first], label)
            first_auc = (score[:first] < score[first]).mean() + 0.5 * (score[:first] == score[first]).mean()
            rows.setdefault(name, []).append((stone_auc, first_auc))
    print(f"stepping stones ({len(next(iter(rows.values())))} held-out broken runs; AUC per run, averaged):")
    for name, values in sorted(rows.items(), key=lambda kv: -np.mean([v[0] for v in kv[1]])):
        values = np.array(values)
        print(f"  {name}: stone AUC {values[:, 0].mean():.2f} (min {values[:, 0].min():.2f}, max {values[:, 0].max():.2f}); "
              f"first success ranked above {values[:, 1].mean():.0%} of earlier saves")


if __name__ == "__main__":
    test = runs(test=True)
    for name in sys.argv[1:] or ["distance", "room", "stone"]:
        globals()[f"{name}_test"](test)
