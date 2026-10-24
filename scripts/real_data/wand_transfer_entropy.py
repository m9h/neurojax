#!/usr/bin/env python
"""Transfer-entropy directed coupling ("who drives whom") among the WAND connectome
harmonics — IDTxl greedy multivariate TE with hierarchical significance testing.

The independent directed-information view of the resting cycle: if the solenoidal
circulation rotates among a few low-order harmonics, the TE network should show a
**directed loop** among them (the rotation direction).  Multivariate conditional TE
removes common-driver / cascade confounds (Novelli et al. 2019).

    PYTHONPATH= WAND_OUT=/data/datasets/wand_src \\
      .venv-oracle/bin/python scripts/real_data/wand_transfer_entropy.py
"""

import os

import numpy as np
import networkx as nx
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")


def main():
    a = np.load(os.path.join(OUT, "harmonic_coeffs.npy"))          # (T, N)
    n = a.shape[1]
    print(f"harmonic coeffs {a.shape}; multivariate TE on {n} connectome harmonics")
    D = Data(a.T, dim_order="ps")                                  # (processes, samples)
    settings = {"cmi_estimator": "JidtGaussianCMI",
                "max_lag_sources": 4, "min_lag_sources": 1,
                "n_perm_max_stat": 200, "n_perm_min_stat": 200,
                "n_perm_omnibus": 200, "n_perm_max_seq": 200, "verbose": False}
    res = MultivariateTE().analyse_network(settings=settings, data=D)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges = []
    for tgt in range(n):
        r = res.get_single_target(tgt, fdr=False)
        te = r.get("omnibus_te", None)
        for s in set(v[0] for v in r["selected_vars_sources"]):
            G.add_edge(s, tgt)
            edges.append((s, tgt, float(te) if te is not None else np.nan))

    print(f"\n=====  Directed coupling among connectome harmonics  =====")
    print(f"{len(edges)} significant directed edges (density "
          f"{len(edges)/(n*(n-1)):.2f})")
    outdeg = dict(G.out_degree()); indeg = dict(G.in_degree())
    drivers = sorted(outdeg, key=lambda k: -outdeg[k])[:4]
    sinks = sorted(indeg, key=lambda k: -indeg[k])[:4]
    print(f"top drivers (out-degree): {[f'H{d+1}({outdeg[d]})' for d in drivers]}")
    print(f"top receivers (in-degree): {[f'H{s+1}({indeg[s]})' for s in sinks]}")
    strong = sorted([e for e in edges if not np.isnan(e[2])], key=lambda e: -e[2])[:6]
    print("strongest directed edges (TE):")
    for s, t, v in strong:
        print(f"    H{s+1:>2d} -> H{t+1:>2d}   TE={v:.4f}")

    cycles = [c for c in nx.simple_cycles(G) if len(c) >= 2]
    print(f"\ndirected cycles among harmonics: {len(cycles)} "
          f"(>=2-node loops = the rotational/circulation structure)")
    for c in sorted(cycles, key=len)[:5]:
        print("    " + " -> ".join(f"H{i+1}" for i in c) + f" -> H{c[0]+1}")
    print("  -> a directed loop among low-order harmonics is the TE signature of the")
    print("     solenoidal cycle (cross-validates the Langevin circulation, who-drives-whom).")


if __name__ == "__main__":
    main()
