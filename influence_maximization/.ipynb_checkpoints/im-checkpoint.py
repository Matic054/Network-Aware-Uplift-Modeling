import torch
import numpy as np
import networkx as nx
from diffusion_models.independent_cascade import optimized_independent_cascade, dmp_ic, ALE_heuristic, modified_ALE, IC_approx_vectorized
from graph_generation.generate_outcome import generate_contagion_outcome
from torch_geometric.data import Data
import heapq

def im_ic(G, k, edge_probs, optimized=True, num_sim=1000):
    S = []
    candidates = list(G.nodes)
    prior_probs = torch.zeros(G.number_of_nodes())
    while len(S) != k:
        best_node = -1
        best_score = -1
        for n in candidates:
            prior_probs[n] = 1
            if optimized:
                ic = optimized_independent_cascade(G, prior_probs, edge_probs, num_sim)
            else:    
                ic = independent_cascade(G, prior_probs, edge_probs, num_sim)
            score = torch.tensor([ic[node] for node in ic]).sum()
            if score > best_score:
                best_score = score
                best_node = n
            prior_probs[n] = 0
        prior_probs[best_node]=1
        candidates.remove(best_node)
        S.append(best_node)
    return S, best_score

def im_diff(G, k, data, diff_model, num_sim=1000):
    S = []
    candidates = list(G.nodes)
    prior_probs = torch.zeros(G.number_of_nodes())
    while len(S) != k:
        best_node = -1
        best_score = -1
        for n in candidates:
            prior_probs[n] = 1
            data.x = prior_probs.unsqueeze(1)
            diff_model.eval()
            with torch.no_grad():
                score = diff_model(data.x, data.edge_index, data.edge_attr).sum()
            if score > best_score:
                best_score = score
                best_node = n
            prior_probs[n] = 0
            data.x = prior_probs.unsqueeze(1)
        prior_probs[best_node]=1
        data.x = prior_probs.unsqueeze(1)
        candidates.remove(best_node)
        S.append(best_node)
    return S, best_score

def im_diff_double(G, k, data, diff_model, num_sim=1000):
    S = []
    candidates = list(G.nodes)
    prior_probs = torch.zeros(G.number_of_nodes())
    while len(S) != k:
        best_node = -1
        best_score = -1
        for n in candidates:
            prior_probs[n] = 1
            data.x = prior_probs.unsqueeze(1)
            diff_model.eval()
            with torch.no_grad():
                new_x = diff_model(data.x, data.edge_index, data.edge_attr)
                score = diff_model(new_x, data.edge_index, data.edge_attr).sum()
            if score > best_score:
                best_score = score
                best_node = n
            prior_probs[n] = 0
            data.x = prior_probs.unsqueeze(1)
        prior_probs[best_node]=1
        data.x = prior_probs.unsqueeze(1)
        candidates.remove(best_node)
        S.append(best_node)
    return S, best_score

def treatment_interference_greedy(G, budget, model, data):
    candidates = list(G.nodes)
    t = np.zeros(G.number_of_nodes())
    num_nodes = G.number_of_nodes()
    while t.sum() != budget:
        best_node = -1
        best_score = -1
        for n in candidates:
            t[n] = 1
            data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
            model.eval()
            with torch.no_grad():
                score = num_nodes-model(data.x, data.edge_index, data.edge_attr).sum()
            if score > best_score:
                best_score = score
                best_node = n
            t[n] = 0
            data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
        t[best_node]=1
        data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
        print("New best score is:",num_nodes-model(data.x, data.edge_index, data.edge_attr).sum())
        candidates.remove(best_node)
        print(f"Running greedy interference. So far picked {t.sum()} nodes out of {budget}.")
    return torch.tensor(t, dtype=torch.float32)

def select_top_baseline_nodes(G, budget, model, data):
    num_nodes = G.number_of_nodes()
    
    # No treatments
    t = np.zeros(num_nodes)
    data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr).detach().cpu().numpy()
    
    # Get indices of top-k highest predictions
    top_indices = np.argsort(-pred.squeeze())[:budget]

    # Mark selected nodes
    t[top_indices] = 1

    print(f"Selected top {budget} nodes based on baseline predictions: {top_indices.tolist()[:10]}, ...")
    return torch.tensor(t, dtype=torch.float32)

def select_top_contagion_nodes(G, edge_dict, budget, model, data, diffusion_method):
    num_nodes = G.number_of_nodes()
    
    # No treatments
    t = np.zeros(num_nodes)
    data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr).detach().cpu().numpy()
        final_pred = generate_contagion_outcome(G, edge_dict, pred, diffusion_method)
    
    # Get indices of top-k highest predictions
    top_indices = np.argsort(-final_pred.squeeze())[:budget]

    # Mark selected nodes
    t[top_indices] = 1

    print(f"Selected top {budget} nodes based on baseline predictions: {top_indices.tolist()[:10]}, ...")
    return torch.tensor(t, dtype=torch.float32)

class CELFNode:
    def __init__(self, node, gain, flag=False):
        self.node = node
        self.gain = gain
        self.flag = flag  # Whether the gain is up to date
    
    def __lt__(self, other):
        # Max-heap based on gain
        return self.gain > other.gain

def treatment_interference_celf(G, budget, model, data):
    num_nodes = G.number_of_nodes()
    t = np.zeros(num_nodes)
    data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

    # Run model once with no treatments to get baseline predictions
    model.eval()
    with torch.no_grad():
        pred_base = model(data.x, data.edge_index, data.edge_attr).detach().cpu()

    queue = []

    # Initial gain: improvement for each node if treated individually
    for n in range(num_nodes):
        t[n] = 1
        data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

        with torch.no_grad():
            pred_treated = model(data.x, data.edge_index, data.edge_attr).detach().cpu()
            gain = pred_base[n].item() - pred_treated[n].item() 
            #gain = pred_treated[n].item() - pred_base[n].item()

        heapq.heappush(queue, CELFNode(n, gain, True))

        t[n] = 0
        data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

    selected = []
    total_selected = 0

    while total_selected < budget:
        top = heapq.heappop(queue)

        if not top.flag:
            # Recompute individual gain
            t[top.node] = 1
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

            with torch.no_grad():
                pred_treated = model(data.x, data.edge_index, data.edge_attr).detach().cpu()
                gain = pred_base[top.node].item() - pred_treated[top.node].item()

            t[top.node] = 0
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
            heapq.heappush(queue, CELFNode(top.node, gain, True))
        else:
            # Accept node
            t[top.node] = 1
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
            selected.append(top.node)
            total_selected += 1
            print(f"Selected {total_selected}/{budget}: Node {top.node} with individual gain {top.gain:.4f}")

    return torch.tensor(t, dtype=torch.float32)

def treatment_contagion_greedy(G, edge_dict, budget, model, data, diffusion_method="ModifiedALE"):
    candidates = list(G.nodes)
    t = np.zeros(G.number_of_nodes())
    num_nodes = G.number_of_nodes()
        
    while t.sum() != budget:
        best_node = -1
        best_score = -1
        for n in candidates:
            t[n] = 1
            data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
            model.eval()
            with torch.no_grad():
                pred_y = model(data.x, data.edge_index, data.edge_attr)
                contagion_y = generate_contagion_outcome(G, edge_dict, pred_y ,diffusion_method)
                score = num_nodes-contagion_y.sum()
            if score > best_score:
                best_score = score
                best_node = n
            t[n] = 0
            data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
        t[best_node]=1
        data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
        candidates.remove(best_node)
        print(f"Running greedy contagion. So far picked {t.sum()} nodes out of {budget}.")
    return torch.tensor(t, dtype=torch.float32)

def treatment_contagion_celf(G, edge_dict, budget, model, data, diffusion_method="ModifiedALE"):
    num_nodes = G.number_of_nodes()
    t = np.zeros(num_nodes)
    data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_base = model(data.x, data.edge_index, data.edge_attr).detach().cpu()

    queue = []

    # Initial gain computation
    for n in range(num_nodes):
        t[n] = 1
        data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred_y = model(data.x, data.edge_index, data.edge_attr)
            contagion_y = generate_contagion_outcome(G, edge_dict, pred_y, diffusion_method)
            #gain = pred_base.sum().item() - contagion_y.sum().item()
            gain = contagion_y.sum().item() - pred_base.sum().item()
        heapq.heappush(queue, CELFNode(n, gain, True))
        t[n] = 0
        data.x[:, 0] = torch.tensor(t, dtype=torch.float32)

    selected = []
    total_selected = 0

    while total_selected < budget:
        top = heapq.heappop(queue)
        
        if not top.flag:
            # Recompute gain
            t[top.node] = 1
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                pred_y = model(data.x, data.edge_index, data.edge_attr)
                contagion_y = generate_contagion_outcome(G, edge_dict, pred_y, diffusion_method)
                gain = num_nodes - contagion_y.sum().item()
            t[top.node] = 0
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
            # Push back with updated gain and flag
            heapq.heappush(queue, CELFNode(top.node, gain, True))
        else:
            # Accept the node
            t[top.node] = 1
            data.x[:, 0] = torch.tensor(t, dtype=torch.float32)
            selected.append(top.node)
            total_selected += 1
            print(f"CELF selected {total_selected}/{budget}: Node {top.node} with gain {top.gain:.4f}")

    return torch.tensor(t, dtype=torch.float32)

def treatment_contagion_greedy_ic(G, edge_dict, budget, model, data, model_ic=None, diffusion_method="ModifiedALE", **kwargs):
    candidates = list(G.nodes)
    t = np.zeros(G.number_of_nodes())
    num_nodes = G.number_of_nodes()
    data.x[:,0] = torch.tensor(t, dtype=torch.float32)#.unsqueeze(1)
    model.eval()
    with torch.no_grad():
        pred_y_base = model(data.x, data.edge_index, data.edge_attr)
    while t.sum() != budget:
        best_node = -1
        best_score = -1
        for n in candidates:
            t[n] = 1

            if diffusion_method=="IC":
                dict_y = optimized_independent_cascade(G, t, edge_dict, k=1000)
                z = np.array([dict_y[id] for id in dict_y])
            elif diffusion_method=="model":
                model = kwargs.get('model', None)
                model.eval()
                with torch.no_grad():
                    z = model(data_t.x, data_t.edge_index, data_t.edge_attr)
            elif diffusion_method=="ALE":
                num_steps = kwargs.get('num_steps', 4)
                z=ALE_heuristic(G, torch.tensor(t, dtype=torch.float32).unsqueeze(1), edge_dict, num_steps)
            elif diffusion_method=="ModifiedALE":
                num_steps = kwargs.get('num_steps', 4)
                z=modified_ALE(G, torch.tensor(t, dtype=torch.float32).unsqueeze(1), edge_dict, num_steps)
            elif diffusion_method=="IC_approx":
                num_steps = kwargs.get('num_steps', 10)
                z=IC_approx_vectorized(G, edge_dict, torch.tensor(t, dtype=torch.float32).unsqueeze(1), num_steps)
            else:
                print("Method not viable!")
                return -1
                
            pred_y = np.maximum(0, pred_y_base-z)
            contagion_y = generate_contagion_outcome(G, edge_dict, pred_y ,diffusion_method,model=model_ic, data=data)
            score = num_nodes-contagion_y.sum()
            if score > best_score:
                best_score = score
                best_node = n
            t[n] = 0
        t[best_node]=1
        candidates.remove(best_node)
        print(f"Running greedy ic contagion. So far picked {t.sum()} nodes out of {budget}.")
    return torch.tensor(t, dtype=torch.float32)
        