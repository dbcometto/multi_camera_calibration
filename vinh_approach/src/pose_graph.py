"""Step 5 — Camera graph: choose reliable camera pairs without ground truth.

Nodes = cameras, edges = pairwise relative poses (from OnlinePair).

1. Candidate edges: enough inliers and a reasonable inlier ratio.
1b. Triangle check: an edge must close at least one rotation-consistent
   triangle with two other edges.
2. Maximum spanning tree on inlier count -> one rotation per camera by
   chaining R_j = R_ij R_i from a root camera.
3. Loop-consistency check: an edge (i, j) is kept only if its own rotation
   agrees with the tree rotations: angle(R_ij, R_j R_i^T) < max_deg.
   Wrong pairs (repeated wall texture can give a confident but wrong E)
   disagree with the rest of the graph and are removed.
"""
import numpy as np

from metrics import rotation_error_deg


def candidate_edges(pairs, min_inliers=150, min_ratio=0.10):
    """pairs: {(a, b): OnlinePair}. Returns list of (a, b) passing the gates."""
    return [k for k, p in pairs.items()
            if p.R is not None and p.n_inliers >= min_inliers and p.inlier_ratio >= min_ratio]


def max_spanning_tree(nodes, edges, weight):
    """Kruskal on -weight. Returns tree edges (a, b) and the connected node set."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for a, b in sorted(edges, key=lambda e: -weight[e]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.append((a, b))
    return tree


def chain_rotations(root, tree, rel_R):
    """BFS from root along tree edges. rel_R[(a,b)] maps a->b (x_b = R x_a)."""
    adj = {}
    for a, b in tree:
        adj.setdefault(a, []).append((b, rel_R[(a, b)]))          # R_b = R_ab R_a
        adj.setdefault(b, []).append((a, rel_R[(a, b)].T))        # R_a = R_ab^T R_b
    R = {root: np.eye(3)}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v, Ruv in adj.get(u, []):
            if v not in R:
                R[v] = Ruv @ R[u]
                queue.append(v)
    return R


def consistent_edges(edges, rel_R, R_abs, max_deg=5.0):
    """Keep edges whose relative rotation matches the chained absolute rotations."""
    kept, rejected = [], []
    for a, b in edges:
        if a not in R_abs or b not in R_abs:
            rejected.append(((a, b), np.inf))
            continue
        err = rotation_error_deg(rel_R[(a, b)], R_abs[b] @ R_abs[a].T)
        (kept if err < max_deg else rejected).append(((a, b), err))
    return kept, rejected


def _rel(rel_R, a, b):
    return rel_R[(a, b)] if (a, b) in rel_R else rel_R[(b, a)].T


def triangle_supported(edges, rel_R, max_deg=5.0):
    """Keep an edge if it closes at least one consistent triangle i-j-k:
    R_ij ≈ R_kj R_ik. A wrong edge almost never agrees with two other
    independent edges, so this filters outliers before the tree is built."""
    es = set(edges)
    nodes = {c for e in edges for c in e}
    has = lambda a, b: (a, b) in es or (b, a) in es  # noqa: E731
    kept = []
    for a, b in edges:
        for k in nodes - {a, b}:
            if has(a, k) and has(k, b):
                loop = _rel(rel_R, k, b) @ _rel(rel_R, a, k)
                if rotation_error_deg(rel_R[(a, b)], loop) < max_deg:
                    kept.append((a, b))
                    break
    return kept


def build_graph(pairs, root=None, min_inliers=150, min_ratio=0.10, max_deg=5.0):
    nodes = sorted({c for k in pairs for c in k})
    cand = candidate_edges(pairs, min_inliers, min_ratio)
    weight = {k: pairs[k].n_inliers for k in cand}
    rel_R = {k: pairs[k].R for k in cand}
    tri = triangle_supported(cand, rel_R, max_deg)
    tree = max_spanning_tree(nodes, tri, weight)
    if root is None:  # camera with the most total inliers on tree edges
        score = {n: sum(weight[e] for e in tree if n in e) for n in nodes}
        root = max(score, key=score.get)
    R_abs = chain_rotations(root, tree, rel_R)
    kept, rejected = consistent_edges(cand, rel_R, R_abs, max_deg)
    return dict(nodes=nodes, candidates=cand, triangle_ok=tri, tree=tree, root=root,
                kept=[k for k, _ in kept], kept_err=dict(kept),
                rejected=dict(rejected), R_chain=R_abs)
