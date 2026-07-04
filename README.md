# Anonymous code

Anonymous code for the paper **Effective and Scalable Algorithms for Influence Estimation and Maximization under Hypergraph Threshold Cascades**.

The repository contains the implementation of HTC influence estimation (IE), influence maximization (IM), RTW-based algorithms, and baselines. A small BlogCatalog example dataset is included for smoke tests. Preprocessing scripts, plotting scripts, paper drafts, and large experiment outputs are intentionally excluded.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Main binaries:

```text
build/htc_im    # influence maximization
build/htc_ie    # influence estimation
```

No third-party C++ libraries are required.

## Input Files

A small example dataset is provided at `data/BlogCatalog`. Each dataset directory should contain:

```text
edges.txt
hyperedges.txt
hyperedge_thresholds.txt
```

`edges.txt`:

```text
# <num_nodes> <num_edges>
u v
u v p
```

Each edge is directed `u -> v`. If `p` is omitted, the program uses `p(u,v)=1/InDeg(v)`.

`hyperedges.txt`:

```text
# <num_hyperedges>
v1 v2 v3 ...
```

`hyperedge_thresholds.txt`:

```text
# <num_hyperedges>
theta_1
theta_2
...
```

The `i`-th threshold corresponds to the `i`-th hyperedge.

## Influence Maximization

Example:

```bash
./build/htc_im \
  --data data/BlogCatalog \
  --method rtw-greedy-p \
  --k 50 \
  --epsilon 0.2 \
  --delta 0.01 \
  --lambda auto \
  --theta0 10000 \
  --mc-simulations 10000 \
  --seed 1 \
  --output result/im.jsonl
```

Supported IM methods:

```text
rtw-greedy, rtw-greedy-p, mc-greedy,
ur-im, ur-im-hist,
edge-only, edge-only-hist,
htc-ce, htc-ce-hist,
hci1-tm, hci2-tm, hn-moea-htc
```

## Influence Estimation

Example:

```bash
./build/htc_ie \
  --data data/BlogCatalog \
  --method rtw-est \
  --seeds-file data/BlogCatalog/IE_seeds/topk_outplushyper_k50.txt \
  --epsilon 0.2 \
  --delta 0.01 \
  --seed 1 \
  --output result/ie.jsonl
```

Alternatively, seeds can be provided directly by `--seeds 1,5,10`.

Supported IE methods:

```text
rtw-est, rtw-fixed, mc-htc, ur-ie
```

## Optional File Overrides

The default input paths are resolved from `--data`. They can be overridden by:

```text
--edges-file <path>
--hyperedges-file <path>
--hyperedge-thresholds-file <path>
```

## Output

Results are written as JSONL to `--output`. A readable companion file with suffix `.readable.txt` is also generated.

Important fields include runtime, peak memory, selected seeds, empirical influence, RTW sample counts, RR sample counts, and certificate-related quantities.
