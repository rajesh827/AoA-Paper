"""
Benchmark harness for the comparative username generation / verification
study — adapted for real datasets supplied as plain text files, one
username per line:

    dataset_10k.txt    (10,000 usernames)
    dataset_100k.txt   (100,000 usernames)
    dataset_1M.txt     (1,000,000 usernames)

The largest file (dataset_1M.txt) is treated as the ceiling: verification
and distributed-deployment experiments sweep across increasing corpus
sizes taken from it. The 10K and 100K files are used directly as
additional, independently-collected checkpoints, since they are real
provided datasets rather than subsets of the 1M file.

Produces:
  results_verification.csv   -- Bloom filter vs trie vs plain-set baseline,
                                 at increasing dataset scale: build time,
                                 memory, query latency, false positive /
                                 negative rate.
  results_distributed.csv    -- consistent hashing vs naive modulo sharding:
                                 fraction of keys remapped when the cluster
                                 grows from N to N+1 nodes.
  results_generation.csv     -- random suffix / Snowflake-style / trie
                                 suggestion / agentic-style generation:
                                 attempts-to-success, latency, readability,
                                 first-guess collision rate.
  results/plots/*.png        -- one chart per axis of comparison.

Run:
  python3 run_benchmark.py
"""

import csv
import time
import tracemalloc
import statistics
import random
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structures import BloomFilter, Trie, ConsistentHashRing, NaiveModuloRing
from generation import (
    random_suffix_generate,
    SnowflakeSuffixGenerator,
    trie_suggest_generate,
    agentic_style_generate,
    readability_score,
)

random.seed(7)

OUT_DIR = "results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATASET_FILES = {
    "10k": "dataset_10k.txt",
    "100k": "dataset_100k.txt",
    "1M": "dataset_1M.txt",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_usernames(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


DATASETS = {label: load_usernames(path) for label, path in DATASET_FILES.items()}

for label, names in DATASETS.items():
    print(f"Loaded dataset '{label}': {len(names)} usernames "
          f"(unique: {len(set(names))})")

# The 1M file is the largest corpus available; it is used as the source
# for the verification and distributed-deployment sweeps, sliced down to
# smaller scales. The full 1M corpus, plus the independently-supplied 10k
# and 100k files, are reported as separate checkpoints.
FULL_CORPUS = DATASETS["1M"]

# Build a candidate request stream: a mix of (a) usernames deliberately
# resampled from the corpus itself (guaranteed collisions, simulating a
# user requesting an already-taken name) and (b) base names derived by
# stripping trailing digits/decorations from real corpus entries
# (simulating what a user actually typed before any suffix was added).
random.seed(11)


def extract_base_name(username):
    """Strip trailing digits/known decorations to approximate what a user
    actually typed before decoration was applied."""
    base = username
    while base and base[-1].isdigit():
        base = base[:-1]
    for sep in ("_", "."):
        if sep in base:
            base = base.split(sep)[0]
    return base or "user"


def build_candidate_stream(corpus, n_requests, reuse_fraction=0.2):
    n_reused = int(n_requests * reuse_fraction)
    n_fresh = n_requests - n_reused
    reused = random.sample(corpus, min(n_reused, len(corpus)))
    fresh_source = random.sample(corpus, min(n_fresh, len(corpus)))
    fresh_bases = [extract_base_name(u) for u in fresh_source]
    requests = list(reused) + fresh_bases
    random.shuffle(requests)
    return requests


CANDIDATE_STREAM = build_candidate_stream(FULL_CORPUS, n_requests=20000)


# ---------------------------------------------------------------------------
# Part 1: Verification benchmark (Bloom filter vs trie vs plain set)
# ---------------------------------------------------------------------------

def time_queries(contains_fn, queries):
    """Return (avg_latency_us, p95_latency_us) over a list of query strings."""
    samples = []
    for q in queries:
        t0 = time.perf_counter()
        contains_fn(q)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)  # microseconds
    samples.sort()
    avg = statistics.mean(samples)
    p95 = samples[int(len(samples) * 0.95) - 1]
    return avg, p95


def run_verification_for_corpus(label, subset, query_sample):
    rows = []
    scale = len(subset)
    subset_set = set(subset)

    # --- plain set (baseline ground-truth structure) ---
    t0 = time.perf_counter()
    tracemalloc.start()
    s = set(subset)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_time = time.perf_counter() - t0
    avg_lat, p95_lat = time_queries(lambda x: x in s, query_sample)
    rows.append(dict(dataset=label, method="plain_set", scale=scale,
                      build_time_s=build_time, peak_memory_kb=peak / 1024,
                      avg_latency_us=avg_lat, p95_latency_us=p95_lat,
                      false_positive_rate=0.0, false_negative_rate=0.0))

    # --- Bloom filter ---
    t0 = time.perf_counter()
    tracemalloc.start()
    bf = BloomFilter(expected_items=scale, target_fp_rate=0.01)
    for u in subset:
        bf.add(u)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_time = time.perf_counter() - t0
    avg_lat, p95_lat = time_queries(lambda x: x in bf, query_sample)

    fp, fn, neg_total, pos_total = 0, 0, 0, 0
    for u in query_sample:
        truth = u in subset_set
        pred = u in bf
        if truth:
            pos_total += 1
            if not pred:
                fn += 1
        else:
            neg_total += 1
            if pred:
                fp += 1
    fp_rate = fp / neg_total if neg_total else 0.0
    fn_rate = fn / pos_total if pos_total else 0.0
    rows.append(dict(dataset=label, method="bloom_filter", scale=scale,
                      build_time_s=build_time, peak_memory_kb=peak / 1024,
                      avg_latency_us=avg_lat, p95_latency_us=p95_lat,
                      false_positive_rate=fp_rate, false_negative_rate=fn_rate))

    # --- Trie ---
    t0 = time.perf_counter()
    tracemalloc.start()
    trie = Trie()
    for u in subset:
        trie.add(u)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    build_time = time.perf_counter() - t0
    avg_lat, p95_lat = time_queries(lambda x: x in trie, query_sample)
    rows.append(dict(dataset=label, method="trie", scale=scale,
                      build_time_s=build_time, peak_memory_kb=peak / 1024,
                      avg_latency_us=avg_lat, p95_latency_us=p95_lat,
                      false_positive_rate=0.0, false_negative_rate=0.0))

    print(f"  verification @ dataset={label} scale={scale} done")
    return rows


def benchmark_verification():
    """
    Runs verification at:
      - the 10k dataset as provided
      - the 100k dataset as provided
      - four increasing slices of the 1M dataset (50k, 100k, 250k, 1M),
        to observe the scaling trend within a single large corpus in
        addition to the two independently-supplied smaller checkpoints.
    """
    rows = []
    query_sample = CANDIDATE_STREAM[:3000]

    rows += run_verification_for_corpus("10k_provided", DATASETS["10k"], query_sample)
    rows += run_verification_for_corpus("100k_provided", DATASETS["100k"], query_sample)

    for scale in [50_000, 100_000, 250_000, 1_000_000]:
        subset = FULL_CORPUS[:scale]
        rows += run_verification_for_corpus(f"1M_sliced_{scale}", subset, query_sample)

    return rows


# ---------------------------------------------------------------------------
# Part 2: Distributed deployment benchmark (consistent hashing vs modulo)
# ---------------------------------------------------------------------------

def benchmark_distributed(scales, start_nodes=8):
    rows = []
    keys_pool = FULL_CORPUS

    for scale in scales:
        keys_sample = keys_pool[:scale]
        node_names = [f"node-{i}" for i in range(start_nodes)]

        # --- consistent hashing ---
        ring = ConsistentHashRing(nodes=node_names, virtual_replicas=100)
        before = {k: ring.get_node(k) for k in keys_sample}
        ring.add_node(f"node-{start_nodes}")  # scale cluster up by one node
        after = {k: ring.get_node(k) for k in keys_sample}
        remapped = sum(1 for k in before if before[k] != after[k])
        frac_remapped_ch = remapped / len(before) if before else 0.0

        # --- naive modulo hashing ---
        modulo_before = NaiveModuloRing(node_names)
        before_m = {k: modulo_before.get_node(k) for k in keys_sample}
        modulo_after = NaiveModuloRing(node_names + [f"node-{start_nodes}"])
        after_m = {k: modulo_after.get_node(k) for k in keys_sample}
        remapped_m = sum(1 for k in before_m if before_m[k] != after_m[k])
        frac_remapped_mod = remapped_m / len(before_m) if before_m else 0.0

        # routing latency (lookup cost per key)
        avg_lat_ch, p95_lat_ch = time_queries(lambda k: ring.get_node(k), keys_sample)
        avg_lat_mod, p95_lat_mod = time_queries(lambda k: modulo_after.get_node(k), keys_sample)

        rows.append(dict(method="consistent_hashing", scale=scale,
                          fraction_remapped=frac_remapped_ch,
                          avg_lookup_latency_us=avg_lat_ch, p95_lookup_latency_us=p95_lat_ch))
        rows.append(dict(method="naive_modulo", scale=scale,
                          fraction_remapped=frac_remapped_mod,
                          avg_lookup_latency_us=avg_lat_mod, p95_lookup_latency_us=p95_lat_mod))

        print(f"  distributed @ scale={scale} done "
              f"(CH remap={frac_remapped_ch:.3f}, modulo remap={frac_remapped_mod:.3f})")

    return rows


# ---------------------------------------------------------------------------
# Part 3: Generation benchmark
# ---------------------------------------------------------------------------

STRATEGY = {
    "random_suffix": "sequential_retry",
    "snowflake_style": "sequential_retry",
    "trie_suggest": "sequential_retry",
    "agentic_style": "batch_evaluate",
}


def run_generation_for_corpus(label, subset, n_requests=300):
    is_taken_set = set(subset)

    trie = Trie()
    for u in subset:
        trie.add(u)

    def is_taken(name):
        return name in is_taken_set

    snowflake_gen = SnowflakeSuffixGenerator(node_id=random.randint(0, 65535))

    sample_source = random.sample(subset, min(n_requests, len(subset)))
    base_names = [extract_base_name(u) for u in sample_source]

    bare_name_taken = sum(1 for bn in base_names if is_taken(bn))
    bare_name_taken_rate = bare_name_taken / len(base_names)

    methods = {
        "random_suffix": lambda bn: random_suffix_generate(bn, is_taken),
        "snowflake_style": lambda bn: snowflake_gen.generate(bn, is_taken),
        "trie_suggest": lambda bn: trie_suggest_generate(bn, trie, is_taken),
        "agentic_style": lambda bn: agentic_style_generate(bn, is_taken),
    }

    rows = []
    for method_name, fn in methods.items():
        strategy = STRATEGY[method_name]
        attempts_list, latencies, readabilities = [], [], []
        successes = 0
        first_guess_collisions = 0

        for bn in base_names:
            t0 = time.perf_counter()
            result, attempts = fn(bn)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)
            attempts_list.append(attempts)
            if result is not None:
                successes += 1
                readabilities.append(readability_score(result))
            if strategy == "sequential_retry" and attempts > 1:
                first_guess_collisions += 1

        rows.append(dict(
            dataset=label,
            method=method_name,
            strategy=strategy,
            corpus_scale=len(subset),
            avg_attempts_or_candidates=statistics.mean(attempts_list),
            success_rate=successes / len(base_names),
            avg_latency_us=statistics.mean(latencies),
            p95_latency_us=sorted(latencies)[int(len(latencies) * 0.95) - 1],
            avg_readability=statistics.mean(readabilities) if readabilities else 0.0,
            first_guess_collision_rate=(first_guess_collisions / len(base_names)
                                         if strategy == "sequential_retry" else None),
            bare_name_taken_rate_shared=bare_name_taken_rate,
        ))
        print(f"  generation @ dataset={label} method={method_name} done")

    return rows


def benchmark_generation():
    """
    Generation is checked at three checkpoints: the two independently
    provided smaller datasets (10k, 100k), and the full 1M dataset, to
    show the scaling trend across two orders of magnitude.
    """
    rows = []
    rows += run_generation_for_corpus("10k_provided", DATASETS["10k"])
    rows += run_generation_for_corpus("100k_provided", DATASETS["100k"])
    rows += run_generation_for_corpus("1M_full", FULL_CORPUS)
    return rows


# ---------------------------------------------------------------------------
# Save + plot helpers
# ---------------------------------------------------------------------------

def save_csv(rows, filename):
    if not rows:
        return
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def plot_verification(rows):
    methods = sorted(set(r["method"] for r in rows))
    scales = sorted(set(r["scale"] for r in rows))

    for metric, ylabel, fname in [
        ("avg_latency_us", "Average query latency (µs)", "latency_vs_scale.png"),
        ("peak_memory_kb", "Peak memory (KB)", "memory_vs_scale.png"),
    ]:
        plt.figure(figsize=(6, 4))
        for m in methods:
            pts = [(r["scale"], r[metric]) for r in rows if r["method"] == m]
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("Existing-username corpus size")
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.title(f"{ylabel} vs. dataset scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
        plt.close()

    plt.figure(figsize=(6, 4))
    for m in methods:
        pts = [(r["scale"], r["false_positive_rate"]) for r in rows if r["method"] == m]
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Existing-username corpus size")
    plt.ylabel("False positive rate")
    plt.xscale("log")
    plt.title("Verification false-positive rate vs. scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "false_positive_rate.png"), dpi=150)
    plt.close()


def plot_distributed(rows):
    methods = sorted(set(r["method"] for r in rows))
    scales = sorted(set(r["scale"] for r in rows))
    plt.figure(figsize=(6, 4))
    for m in methods:
        ys = [next(r["fraction_remapped"] for r in rows if r["method"] == m and r["scale"] == s)
              for s in scales]
        plt.plot(scales, ys, marker="o", label=m)
    plt.xlabel("Number of keys routed")
    plt.ylabel("Fraction of keys remapped after adding 1 node")
    plt.title("Remapping overhead: consistent hashing vs. naive modulo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "remap_fraction.png"), dpi=150)
    plt.close()


def plot_generation(rows):
    datasets = sorted(set(r["dataset"] for r in rows), key=lambda d: rows_scale(rows, d))
    methods = sorted(set(r["method"] for r in rows))

    for metric, ylabel, fname in [
        ("avg_attempts_or_candidates", "Avg attempts / candidates evaluated", "gen_attempts.png"),
        ("avg_latency_us", "Average latency per request (µs)", "gen_latency.png"),
        ("avg_readability", "Average readability score (heuristic, 0-1)", "gen_readability.png"),
        ("success_rate", "Success rate", "gen_success_rate.png"),
    ]:
        plt.figure(figsize=(7, 4))
        width = 0.2
        x = range(len(datasets))
        for i, m in enumerate(methods):
            ys = [next(r[metric] for r in rows if r["method"] == m and r["dataset"] == d)
                  for d in datasets]
            offset = (i - len(methods) / 2) * width
            plt.bar([xi + offset for xi in x], ys, width=width, label=m)
        plt.xticks(list(x), datasets)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by dataset")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
        plt.close()

    retry_rows = [r for r in rows if r["first_guess_collision_rate"] is not None]
    if retry_rows:
        plt.figure(figsize=(7, 4))
        width = 0.25
        retry_methods = sorted(set(r["method"] for r in retry_rows))
        x = range(len(datasets))
        for i, m in enumerate(retry_methods):
            ys = [next(r["first_guess_collision_rate"] for r in retry_rows
                       if r["method"] == m and r["dataset"] == d) for d in datasets]
            offset = (i - len(retry_methods) / 2) * width
            plt.bar([xi + offset for xi in x], ys, width=width, label=m)
        plt.xticks(list(x), datasets)
        plt.ylabel("First-guess collision rate")
        plt.title("First-guess collision rate (sequential-retry methods only)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "gen_first_guess_collision_rate.png"), dpi=150)
        plt.close()


def rows_scale(rows, dataset_label):
    return next(r["corpus_scale"] for r in rows if r["dataset"] == dataset_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running verification benchmark...")
    verification_rows = benchmark_verification()
    save_csv(verification_rows, "results_verification.csv")
    plot_verification(verification_rows)

    print("Running distributed-deployment benchmark...")
    distributed_rows = benchmark_distributed(scales=[5_000, 20_000, 50_000, 100_000])
    save_csv(distributed_rows, "results_distributed.csv")
    plot_distributed(distributed_rows)

    print("Running generation benchmark...")
    generation_rows = benchmark_generation()
    save_csv(generation_rows, "results_generation.csv")
    plot_generation(generation_rows)

    print("\nAll benchmarks complete. Results in ./results/")
