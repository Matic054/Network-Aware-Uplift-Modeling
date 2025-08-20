import networkx as nx
import igraph as ig
import numpy as np
import torch
from diffusion_models.edge_probabilities import generate_edge_probabilities
from models.edge_gnn_models import GraphEdgeModel

def generate_erdos_renyi_graph(num_nodes, edge_prob, prob_selected = 0.3, max_prior=1, seed = None):
    """
    Generate a random graph and initialize prior probabilities.
    
    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - edge_prob (float): Probability of edge creation between nodes.
    
    Returns:
    - G (networkx.Graph): Generated graph.
    - prior_probs (torch.Tensor): Tensor of prior probabilities for each node.
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            #import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        G_undirected = nx.erdos_renyi_graph(num_nodes, edge_prob)
        G = nx.DiGraph(seed=seed)
    else:
        G_undirected = nx.erdos_renyi_graph(num_nodes, edge_prob)
        G = nx.DiGraph()

    for u in G_undirected.nodes():
            G.add_node(u)

    for u, v in G_undirected.edges():
            G.add_edge(u, v)
            G.add_edge(v, u)
    #G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    #G = nx.gnp_random_graph(num_nodes, edge_prob)

    # Step 1: Create a mask for selecting 30% of the entries
    mask = torch.rand(num_nodes) < prob_selected  # Random selection based on prob_selected
    
    # Step 2: Assign random values between 0 and 0.5 for the selected entries
    #prior_probs = torch.rand(num_nodes) * mask.float() * max_prior 
    prior_probs = mask.float()
    return G, prior_probs

def generate_erdos_renyi_attribute_graph(num_nodes,
                                              edge_prob,
                                              prob_selected=0.3,
                                              max_prior=1,
                                              num_node_features=1,
                                              num_edge_features=1,
                                              node_weights=None,
                                              edge_weights=None,
                                              node_weights_source=None,
                                              node_weights_sink=None,
                                              epsilon=None,
                                              edge_method='weighted_sum',
                                              max_edge=0.333,
                                              b_s=0.333,
                                              b_t=0.333,
                                              b_e=0.333,
                                              attribute_distribution="uniform",
                                              max_node_features=1,
                                              provided_graph=None,
                                              both_dir=True):
    """
    Generate a BA graph with node and edge features.
    One of the node features is the prior infection probability.

    Returns:
    - G (networkx.Graph): Graph with node and edge features.
    - prior_probs (torch.Tensor): Tensor of prior infection probabilities.
    """
    if provided_graph is None:
        G_undirected = nx.erdos_renyi_graph(num_nodes, edge_prob)
        G = nx.DiGraph()
    
        for node in G_undirected.nodes():
            G.add_node(node)
    
        for u, v in G_undirected.edges():
            G.add_edge(u, v)
            G.add_edge(v, u)
    else:
        G = provided_graph

    # Create mask and assign prior infection probabilities
    mask = torch.rand(num_nodes) < prob_selected
    prior_probs = torch.rand(num_nodes) * mask.float() * max_prior

    # Add node features
    for node in G.nodes():
        if num_node_features > 0:
            if attribute_distribution=="uniform":
                other_features = np.random.rand(num_node_features - 1) * max_node_features
            elif attribute_distribution=="normal":
                other_features = np.random.normal(size=num_node_features - 1) * max_node_features
            full_features = np.concatenate([[prior_probs[node].item()], other_features])
            G.nodes[node]['features'] = full_features
        else:
            G.nodes[node]['features'] = np.array([0])

    # Initialize random weights if not provided
    if node_weights is None:
        node_weights = np.random.rand(num_node_features)
    if edge_weights is None:
        edge_weights = np.random.rand(num_edge_features)

    # Add edge features
    for (u,v) in G.edges():
        if num_edge_features > 0:
            if attribute_distribution=="uniform":
                G.edges[(u,v)]['features'] = np.random.rand(num_edge_features) * max_node_features
                if both_dir:
                    G.edges[(v,u)]['features'] = G.edges[(u,v)]['features']
            elif attribute_distribution=="normal":
                G.edges[(u,v)]['features'] = np.random.normal(size=num_edge_features) * max_node_features
                if both_dir:
                    G.edges[(v,u)]['features'] = G.edges[(u,v)]['features']
        else:
            G.edges[(u,v)]['features'] = np.array([0])
            G.edges[(v,u)]['features'] = np.array([0])

    # Generate edge probabilities
    if edge_method == "gnn":
        model = GraphEdgeModel(node_in_dim=G.nodes[0]['features'].shape[0],
                       edge_in_dim=list(G.edges(data=True))[0][2]['features'].shape[0])
    else: 
        model = -1
    edge_dict = generate_edge_probabilities(G,
                                            method=edge_method,
                                            node_weights=node_weights,
                                            edge_weights=edge_weights,
                                            max_edge=max_edge,
                                            node_weights_source=node_weights_source,
                                            node_weights_sink=node_weights_sink,
                                            epsilon=epsilon,
                                            model = model,
                                            both_dir = both_dir)

    return G, prior_probs, edge_dict


def generate_barabasi_albert_graph(num_nodes, num_edges_per_node, prob_selected=0.3, max_prior=1, seed=None, use_preloaded_graph=False, preloaded_graph=None):
    """
    Generate a Barabási–Albert (BA) graph and initialize prior probabilities.

    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - num_edges_per_node (int): Number of edges to attach from a new node to existing nodes.
    - prob_selected (float): Fraction of nodes initially infected (prior probs).

    Returns:
    - G (networkx.Graph): Generated BA graph.
    - prior_probs (torch.Tensor): Tensor of prior infection probabilities.
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            #import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        if not use_preloaded_graph:
            G_undirected = nx.barabasi_albert_graph(num_nodes, num_edges_per_node, seed=seed)
            G = nx.DiGraph(seed=seed)
    else:
        if not use_preloaded_graph:
            G_undirected = nx.barabasi_albert_graph(num_nodes, num_edges_per_node)
            G = nx.DiGraph()

    if not use_preloaded_graph:
        for u, v in G_undirected.edges():
            G.add_edge(u, v)
            G.add_edge(v, u)

    # Create mask and assign prior infection probabilities
    if use_preloaded_graph:
        G = preloaded_graph
    mask = torch.rand(num_nodes) < prob_selected
    prior_probs = mask.float() * max_prior * torch.rand(num_nodes)

    return G, prior_probs

def generate_barabasi_albert_attribute_graph(num_nodes,
                                              num_edges_per_node,
                                              prob_selected=0.3,
                                              max_prior=1,
                                              num_node_features=1,
                                              num_edge_features=1,
                                              node_weights=None,
                                              edge_weights=None,
                                              node_weights_source=None,
                                              node_weights_sink=None,
                                              epsilon=None,
                                              edge_method='weighted_sum',
                                              max_edge=0.333,
                                              b_s=0.333,
                                              b_t=0.333,
                                              b_e=0.333,
                                              attribute_distribution="uniform",
                                              max_node_features=1):
    """
    Generate a BA graph with node and edge features.
    One of the node features is the prior infection probability.

    Returns:
    - G (networkx.Graph): Graph with node and edge features.
    - prior_probs (torch.Tensor): Tensor of prior infection probabilities.
    """
    G_undirected = nx.barabasi_albert_graph(num_nodes, num_edges_per_node)
    G = nx.DiGraph()

    for u, v in G_undirected.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    # Create mask and assign prior infection probabilities
    mask = torch.rand(num_nodes) < prob_selected
    prior_probs = torch.rand(num_nodes) * mask.float() * max_prior

    # Add node features
    for node in G.nodes():
        if num_node_features > 0:
            if attribute_distribution=="uniform":
                other_features = np.random.rand(num_node_features - 1) * max_node_features
            elif attribute_distribution=="normal":
                other_features = np.random.normal(size=num_node_features - 1) * max_node_features
            full_features = np.concatenate([[prior_probs[node].item()], other_features])
            G.nodes[node]['features'] = full_features
        else:
            G.nodes[node]['features'] = np.array([0])

    # Initialize random weights if not provided
    if node_weights is None:
        node_weights = np.random.rand(num_node_features)
    if edge_weights is None:
        edge_weights = np.random.rand(num_edge_features)

    # Add edge features
    for (u,v) in G.edges():
        if num_edge_features > 0:
            if attribute_distribution=="uniform":
                G.edges[(u,v)]['features'] = np.random.rand(num_edge_features) * max_node_features
                G.edges[(v,u)]['features'] = G.edges[(u,v)]['features']
            elif attribute_distribution=="normal":
                G.edges[(u,v)]['features'] = np.random.normal(size=num_edge_features) * max_node_features
                G.edges[(v,u)]['features'] = G.edges[(u,v)]['features']
        else:
            G.edges[(u,v)]['features'] = np.array([0])
            G.edges[(v,u)]['features'] = np.array([0])

    # Generate edge probabilities
    if edge_method == "gnn":
        model = GraphEdgeModel(node_in_dim=G.nodes[0]['features'].shape[0],
                       edge_in_dim=list(G.edges(data=True))[0][2]['features'].shape[0])
    else: 
        model = -1
    edge_dict = generate_edge_probabilities(G,
                                            method=edge_method,
                                            node_weights=node_weights,
                                            edge_weights=edge_weights,
                                            max_edge=max_edge,
                                            node_weights_source=node_weights_source,
                                            node_weights_sink=node_weights_sink,
                                            epsilon=epsilon,
                                            model = model)

    return G, prior_probs, edge_dict