"""
Builds the 3-partition spatial configuration adjacency matrix for ST-GCN,
as described in Yan et al. (2018), for the COCO-17 skeleton.

Partitions:
  0 — self-connections
  1 — centripetal neighbours  (neighbour closer to root than current node)
  2 — centrifugal neighbours  (neighbour farther from root than current node)

Root is left_hip (joint 11) — the most stable joint during gait.
Returns A of shape (3, 17, 17), D^{-1/2} A D^{-1/2} normalised per partition.
"""

import numpy as np
from collections import defaultdict, deque

from src.skeleton import NUM_JOINTS, EDGES


def _bfs_distances(edges, root, num_nodes):
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    dist = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def _normalize(A):
    D = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_inv_sqrt = np.where(D > 0, D ** -0.5, 0.0)
    return np.diag(D_inv_sqrt) @ A @ np.diag(D_inv_sqrt)


def build_adjacency(num_joints: int = NUM_JOINTS,
                    edges: list = EDGES,
                    root: int = 11) -> np.ndarray:
    # COCO-17 head joints (0-4) have no path to the body in the base edge list.
    # Add nose→shoulder edges so BFS from hip can reach the full skeleton.
    edges = list(edges) + [(0, 5), (0, 6)]

    dist = _bfs_distances(edges, root, num_joints)

    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)

    # partition 0: self-loops
    for i in range(num_joints):
        A[0, i, i] = 1.0

    # partitions 1 (centripetal) and 2 (centrifugal)
    for i, j in edges:
        if dist[i] > dist[j]:       # j is closer to root → j is centripetal for i
            A[1, i, j] = 1.0        # i receives from j (centripetal)
            A[2, j, i] = 1.0        # j receives from i (centrifugal)
        elif dist[i] < dist[j]:     # i is closer to root
            A[1, j, i] = 1.0
            A[2, i, j] = 1.0
        else:                       # same depth — place in centripetal
            A[1, i, j] = 1.0
            A[1, j, i] = 1.0

    for k in range(3):
        A[k] = _normalize(A[k])

    return A
