import numpy as np
import networkx as nx
import torch

def generate_edge_probabilities(G, method='uniform', seed=None, **kwargs):
    """
    Generate edge probabilities for the graph G based on the specified method.

    Parameters:
    - G (networkx.Graph): The input graph.
    - method (str): The method to use for generating probabilities. Options:
        - 'uniform': Assign a uniform probability to all edges.
        - 'random': Assign random probabilities to edges.
        - 'weighted': Assign probabilities based on edge weights.
    - kwargs: Additional parameters specific to the chosen method.

    Returns:
    - edge_probs (dict): A dictionary where keys are edges and values are probabilities.
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
    edge_probs = {}

    if method == 'uniform':
        p = kwargs.get('p', 0.01)
        for edge in G.edges():
            edge_probs[edge] = p
            if G.is_directed()==False:
                edge_probs[edge] = p

    elif method == 'random':
        low = kwargs.get('low', 0.01)
        high = kwargs.get('high', 1)
        for (u, v) in G.edges():
            edge_probs[(u, v)] = np.random.uniform(low, high)
            #edge_probs[(u,v)] = np.random.normal(loc=0.5, scale=0.25)/3
            if edge_probs[(u,v)] < 0:
                edge_probs[(u,v)] = 0
            elif edge_probs[(u,v)] > 1:
                edge_probs[(u,v)] = 1
            edge_probs[(v, u)] = edge_probs[(u, v)]
            #if G.is_directed()==False:
                #edge_probs[(v, u)] = edge_probs[(u, v)]

    elif method == 'weighted_sum':
        # Retrieve or set default weights for node and edge features
        node_weights = kwargs.get('node_weights', None)
        edge_weights = kwargs.get('edge_weights', None)
        max_edge = kwargs.get('max_edge', 0.333)

        for v, u, data in G.edges(data=True):
            # Retrieve node features; assume they are stored under the 'features' key
            v_features = np.array(G.nodes[v].get('features', np.zeros(1)))
            u_features = np.array(G.nodes[u].get('features', np.zeros(1)))
            # Retrieve edge features; assume they are stored under the 'features' key
            e_features = np.array(data.get('features', np.zeros(1)))

            # Validate or initialize weights
            if node_weights is None:
                node_weights = np.ones_like(v_features)
            if edge_weights is None:
                edge_weights = np.ones_like(e_features)

            # Compute weighted sums
            node_feature_sum = np.dot(node_weights, v_features) + np.dot(node_weights, u_features)
            edge_feature_sum = np.dot(edge_weights, e_features)
            edge_score = (node_feature_sum + edge_feature_sum)

            # Make sigmoid input more negative (optional)
            # edge_score = -abs(edge_score)  # stronger push toward 0
        
            #edge_probs[(v, u)] = 1 / (1 + np.exp(-edge_score))
            # Step 1: Get all the values
            degree = max(G.out_degree(u), 1)
            edge_probs[(v, u)] = edge_score/degree
            if G.is_directed()==False:
                edge_probs[(u, v)] = edge_score
        # Step 1: Get all the values
        values = list(edge_probs.values())
        
        # Step 2: Compute min and max
        min_val = min(values)
        max_val = max(values)
        
        # Step 3: Normalize using the formula
        edge_probs = {
            edge: max_edge*(prob - min_val) / (max_val - min_val)
            #edge: max_edge / (1 + np.exp(prob))
            for edge, prob in edge_probs.items()
        }
    elif method == 'weighted_sum_squares':
        # Retrieve or set default weights for node and edge features
        node_weights = kwargs.get('node_weights', None)
        node_weights_2 = kwargs.get('node_weights_2', None)
        edge_weights = kwargs.get('edge_weights', None)
        edge_weights_2 = kwargs.get('edge_weights_2', None)
        max_edge = kwargs.get('max_edge', 0.333)

        for v, u, data in G.edges(data=True):
            # Retrieve node features; assume they are stored under the 'features' key
            v_features = np.array(G.nodes[v].get('features', np.zeros(1)))
            u_features = np.array(G.nodes[u].get('features', np.zeros(1)))
            # Retrieve edge features; assume they are stored under the 'features' key
            e_features = np.array(data.get('features', np.zeros(1)))

            # Validate or initialize weights
            if node_weights is None:
                node_weights = np.ones_like(v_features)
            if node_weights_2 is None:
                node_weights_2 = np.ones_like(v_features)
            if edge_weights is None:
                edge_weights = np.ones_like(e_features)
            if edge_weights_2 is None:
                edge_weights_2 = np.ones_like(e_features)

            # Compute weighted sums
            node_feature_sum = np.dot(node_weights, v_features) #+ np.dot(node_weights, u_features)
            node_feature_sum_2 = np.dot(node_weights, v_features**2)
            edge_feature_sum = np.dot(edge_weights, e_features)
            edge_feature_sum_2 = np.dot(edge_weights, e_features**2)
            edge_score = (node_feature_sum + node_feature_sum_2 + edge_feature_sum + edge_feature_sum_2)

            # Make sigmoid input more negative (optional)
            # edge_score = -abs(edge_score)  # stronger push toward 0
        
            #edge_probs[(v, u)] = 1 / (1 + np.exp(-edge_score))
            # Step 1: Get all the values
            edge_probs[(v, u)] = edge_score
            if G.is_directed()==False:
                edge_probs[(u, v)] = edge_score
        # Step 1: Get all the values
        values = list(edge_probs.values())
        
        # Step 2: Compute min and max
        min_val = min(values)
        max_val = max(values)
        
        # Step 3: Normalize using the formula
        edge_probs = {
            edge: max_edge*(prob - min_val) / (max_val - min_val)
            for edge, prob in edge_probs.items()
        }
    elif method == 'degree_based':
        c = kwargs.get('c', 1)  # overall scaling factor
        for v, u in G.edges():
            if G.is_directed():
                degree = max(G.out_degree(u), 1)  # Avoid division by 0
                edge_probs[(v, u)] = c / degree
            else:
                degree = max(G.degree(v), 1)  # Avoid division by 0
                edge_probs[(v, u)] = c / degree
                edge_probs[(u, v)] = edge_probs[(v, u)]
            
    elif method == 'neighbor_sum_features':
        # Add a signal from 2-hop away nodes
        for v, u in G.edges():
            v_features = G.nodes[v]['features']
            u_features = G.nodes[u]['features']
            edge_features = G.edges[v,u]['features']

            node_weights = kwargs.get('node_weights', 1)
            edge_weights = kwargs.get('edge_weights', 1)
            max_edge = kwargs.get('max_edge', 0.333)
            
            two_hop_score = 0
            for neighbor in G.neighbors(v):
                if neighbor != u:
                    two_hop_score += np.dot(v_features, G.nodes[neighbor]['features'])
            for neighbor in G.neighbors(u):
                if neighbor != v:
                    two_hop_score += np.dot(u_features, G.nodes[neighbor]['features'])
            
            edge_score = (
                np.dot(v_features, u_features) +
                np.dot(edge_weights, G.edges[v, u]['features']) +
                two_hop_score  
            )
            edge_probs[(v, u)] = max_edge / (1 + np.exp(-edge_score))
            if G.is_directed()==False:
                edge_probs[(u, v)] = edge_probs[(v, u)]

    elif method == 'sigmoid':
        # Retrieve or set default weights for node and edge features
        node_weights_source = kwargs.get('node_weights_source', None)
        node_weights_sink = kwargs.get('node_weights_sink', None)
        edge_weights = kwargs.get('edge_weights', None)
        epsilon = kwargs.get('epsilon', None)
        max_edge = kwargs.get('max_edge', 0.333)
        b_s = kwargs.get('b_s', 0.333)
        b_t = kwargs.get('b_t', 0.333)
        b_e = kwargs.get('b_e', 0.333)

        for v, u, data in G.edges(data=True):
            # Retrieve node features; assume they are stored under the 'features' key
            v_features = np.array(G.nodes[v].get('features', np.zeros(1)))
            u_features = np.array(G.nodes[u].get('features', np.zeros(1)))
            # Retrieve edge features; assume they are stored under the 'features' key
            e_features = np.array(data.get('features', np.zeros(1)))

            # Validate or initialize weights
            if node_weights_source is None:
                node_weights_source = np.ones_like(v_features)
            if node_weights_sink is None:
                node_weights_sink = np.ones_like(u_features)
            if edge_weights is None:
                edge_weights = np.ones_like(e_features)
            if epsilon is None:
                epsilon = np.array(0.01)

            # Compute weighted sums
            source_contribution = 1/(1+np.exp(-np.dot(node_weights_source, v_features)))
            sink_contribution = 1/(1+np.exp(-np.dot(node_weights_sink, u_features)))
            edge_contribution = 1/(1+np.exp(-np.exp(np.dot(edge_weights, e_features))))
            edge_score = b_s*source_contribution + b_t*sink_contribution + b_e*edge_contribution + epsilon
            #node_feature_sum = np.dot(node_weights_source, v_features) + np.dot(node_weights_sink, u_features)
            #edge_feature_sum = np.dot(edge_weights, e_features)
            #edge_score = (node_feature_sum + edge_feature_sum+epsilon)
        
            edge_probs[(v, u)] = max_edge / (1 + np.exp(-edge_score))
            if G.is_directed()==False:
                edge_probs[(u, v)] = edge_probs[(v, u)]
    elif method == 'gnn':
        # Step 1: Index nodes and build feature matrix
        node_features = []
        node_index_map = {}
        for i, node in enumerate(G.nodes()):
            node_index_map[node] = i
            features = np.array(G.nodes[node].get('features', np.zeros(1)))
            node_features.append(features)
        x = torch.tensor(np.stack(node_features), dtype=torch.float)
    
        # Step 2: Build edge_index and edge_attr
        edge_list = []
        edge_attr_list = []
        edge_tuples = []
    
        for v, u, data in G.edges(data=True):
            edge_list.append((node_index_map[v], node_index_map[u]))
            edge_attr = np.array(data.get('features', np.zeros(1)))
            edge_attr_list.append(edge_attr)
            edge_tuples.append((v, u))
    
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float)
    
        # Step 3: Move to model's device (if needed)
        model = kwargs.get('model', -1)
        if model == -1:
            return -1
        device = next(model.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
    
        # Step 4: Forward pass through the model
        model.eval()
        with torch.no_grad():
            edge_outputs = model(x, edge_index, edge_attr)
    
        # Step 5: Build edge_probs dict { (v, u): prob }
        edge_probs = {
            (v, u): prob.item() for (v, u), prob in zip(edge_tuples, edge_outputs)
        }
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'random', or 'weighted'.")

    return edge_probs
