import numpy as np
import torch
import networkx as nx
import math
from collections import defaultdict
import itertools
from scipy.sparse import csr_matrix

def optimized_independent_cascade(G, prior_probs, edge_probs, k, seed=None):
    """
    Optimized version of the Independent Cascade model using NumPy.

    Parameters:
    - G (networkx.Graph): The input graph.
    - prior_probs (dict): Initial infection probabilities for each node.
    - edge_probs (dict): Activation probabilities for each edge.
    - k (int): Number of simulation steps.

    Returns:
    - posterior_probs (dict): Infection probabilities after k steps.
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    nodes = list(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_indices.items()}
    n = len(nodes)

    # Precompute adjacency list with edge probabilities
    adjacency = {node: [] for node in nodes}
    for u, v in G.edges():
        adjacency[u].append((v, edge_probs[(u, v)]))
        #if (u, v) in edge_probs:
            #adjacency[u].append((v, edge_probs[(u, v)]))
        #if (v, u) in edge_probs:
            #adjacency[v].append((u, edge_probs[(v, u)]))

    infection_counts = np.zeros(n, dtype=np.int32)
    prior_array=np.array(prior_probs.squeeze())

    for _ in range(k):
        rand_vals = np.random.rand(n)
        active = rand_vals < prior_array
        visited = active.copy()
        infection_counts += active.astype(np.int32)

        newly_active = set(np.where(active)[0])

        while newly_active:
            next_active = set()
            for idx in newly_active:
                node = idx_to_node[idx]
                for neighbor, p in adjacency[node]:
                    neighbor_idx = node_indices[neighbor]
                    if not visited[neighbor_idx] and np.random.rand() < p:
                        visited[neighbor_idx] = True
                        next_active.add(neighbor_idx)
                        infection_counts[neighbor_idx] += 1
            newly_active = next_active

    posterior_probs = {idx_to_node[i]: infection_counts[i] / k for i in range(n)}
    return posterior_probs

def dmp_ic(G, prior_probs, edge_probs, T=10):
    """
    Dynamic Message Passing (DMP) for influence estimation in Independent Cascade model.
    
    Based on Algorithm 1 from: https://arxiv.org/pdf/1912.12749.pdf

    Parameters:
    - G (networkx.DiGraph): Directed graph
    - prior_probs (dict): Initial infection probabilities for each node
    - edge_probs (dict): Transmission probabilities for each edge (i, j)
    - T (int): Number of DMP iterations

    Returns:
    - pi (dict): Estimated infection probability of each node
    - total_influence (float): Sum of pi values (expected number of infected nodes)
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_node = {idx: node for node, idx in node_idx.items()}

    A = np.zeros((n, n))
    P = np.zeros((n, n))  # edge activation matrix

    # Build adjacency and probability matrices
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        A[i, j] = 1
        P[i, j] = edge_probs.get((u, v), 0.0)

    p0 = np.zeros(n)
    for node in nodes:
        p0[node_idx[node]] = float(prior_probs[node])

    pi = p0.copy()

    # Initialize messages
    p_ij = np.tile(pi, (n, 1)).T  # shape (n, n)

    edges_idx = np.transpose(np.nonzero(A.T))
    for t in range(T):
        q = 1 - P * p_ij.T
        q_ = np.prod(q, axis=1)
        qq = (1 - p0) * q_

        for (i, j) in edges_idx:
            if q[j, i] != 0:
                p_ij[j, i] = 1 - qq[j] / q[j, i]

    # Final node infection probabilities
    q = 1 - P * p_ij.T
    pi = 1 - (1 - p0) * np.prod(q, axis=1)

    #pi_dict = {idx_node[i]: pi[i] for i in range(n)}
    #total_influence = float(np.sum(pi))

    return pi#, total_influence

def ALE_heuristic(G, prior_probs, edge_probs, num_steps):
    """
    Function for estimnating the independent cascade based on ALE model from:
    https://doi.org/10.14232/actacyb.21.1.2013.4

    
    G: networkx.DiGraph
    prior_probs: (num_nodes,) array of initial infection probabilities
    edge_probs: dict mapping (u, v) -> infection probability
    num_steps: int, number of diffusion steps
    """
    # Map nodes to indices
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # Build edge_index and edge_probs tensor
    edge_src = []
    edge_dst = []
    edge_p_list = []

    for u, v in G.edges():
        edge_src.append(node_to_idx[u])
        edge_dst.append(node_to_idx[v])
        edge_p_list.append(edge_probs.get((u, v), 0.0))  # default to 0 if missing

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)  # (2, E)
    edge_probs_tensor = torch.tensor(edge_p_list, dtype=torch.float32)  # (E,)

    # Initial infection vector
    x = torch.tensor(prior_probs, dtype=torch.float32).view(-1, 1)  # (N, 1)

    # Run diffusion: y = x + Bx + B^2x + ... (no weights)
    current_x = x.clone()
    result = x.clone()  # first term: I*x

    src, dst = edge_index
    for _ in range(1, num_steps):
        # Propagate current_x through B = A^T
        edge_vals = current_x[src] * edge_probs_tensor.view(-1, 1)
        current_x = torch.zeros_like(x).scatter_add_(0, dst.view(-1, 1).expand(-1, 1), edge_vals)
        result += current_x

    return result.view(-1)#, node_list 

def modified_ALE(G, prior_probs, edge_probs, num_steps):
    """
    Improved function for estimnating the independent cascade based on ALE model from:
    https://doi.org/10.14232/actacyb.21.1.2013.4

    
    G: networkx.DiGraph
    prior_probs: (num_nodes,) array of initial infection probabilities
    edge_probs: dict mapping (u, v) -> infection probability
    num_steps: int, number of propagation steps
    """
    # Map nodes to indices
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # Build edge_index and edge_probs tensor
    edge_src = []
    edge_dst = []
    edge_p_list = []

    for u, v in G.edges():
        edge_src.append(node_to_idx[u])
        edge_dst.append(node_to_idx[v])
        edge_p_list.append(edge_probs.get((u, v), 0.0))  # default to 0 if missing

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_probs_tensor = torch.tensor(edge_p_list, dtype=torch.float32)

    # Initial infection vector
    x = torch.tensor(prior_probs, dtype=torch.float32).view(-1, 1)

    src, dst = edge_index
    num_nodes = x.size(0)

    # Start with survival = 1 - x (initial infection chance)
    survival_prob = 1 - x

    current_x = x.clone()

    for _ in range(num_steps):
        # Message passing: B * current_x
        messages = current_x[src] * edge_probs_tensor.view(-1, 1)
        current_x = torch.zeros_like(x).scatter_add_(0, dst.view(-1, 1).expand(-1, 1), messages)

        # Update survival: multiply by (1 - current infection input)
        survival_prob = survival_prob * (1 - current_x)

    # Final infection = 1 - survival
    final_infection = 1 - survival_prob

    return final_infection.view(-1)#, node_list

def IC_approx_vectorized(G, edge_probs, prior_probs, T, a):
    n = G.number_of_nodes()
    node_list = list(G.nodes())
    node_index = {u: i for i, u in enumerate(node_list)}

    # Build sparse weighted adjacency matrix A (shape: n x n)
    row, col, data = [], [], []
    for (u, v), p in edge_probs.items():
        row.append(node_index[v])  # Target node (influence into v)
        col.append(node_index[u])  # Source node
        data.append(p)
    A = csr_matrix((data, (row, col)), shape=(n, n))  # A[v, u] = p_{uv}

    # Initialize P matrix: shape (k+1, n)
    P = np.zeros((T + 1, n))
    P[0] = prior_probs

    product_term = np.ones(n)

    for t in range(1, T + 1):
        delta = A.dot(P[t - 1] * a)
        influence_term = np.exp(-(np.clip(delta, a_min=0, a_max=10)))
        product_term *= (1 - P[t - 1])  # Incremental update
        P[t] = product_term * (1 - influence_term)

    final_probs = 1 - np.prod(1 - P, axis=0)
    return final_probs

def IC_approx_vectorized_torch(G, edge_probs, prior_probs, k):
    n = G.number_of_nodes()
    node_list = list(G.nodes())
    node_index = {u: i for i, u in enumerate(node_list)}

    # Prepare indices and values for sparse matrix
    row, col, data = [], [], []
    for (u, v), p in edge_probs.items():
        row.append(node_index[v])  # Target (row)
        col.append(node_index[u])  # Source (col)
        data.append(p)
    
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.tensor(data, dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices, values, (n, n))

    # Initialize P matrix (k+1, n) with prior_probs at t=0
    P = torch.zeros((k + 1, n), dtype=torch.float32)
    P[0] = torch.tensor(prior_probs, dtype=torch.float32)

    for t in range(1, k + 1):
        product_term = torch.prod(1 - P[:t], dim=0)

        prev_P = P[t - 1]
        delta = torch.sparse.mm(A, prev_P.unsqueeze(1)).squeeze(1)

        influence_term = torch.exp(-delta)
        P[t] = product_term * (1 - influence_term)

    final_probs = 1 - torch.prod(1 - P, dim=0)
    return final_probs