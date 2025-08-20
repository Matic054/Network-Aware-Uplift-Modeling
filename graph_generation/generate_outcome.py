import networkx as nx
import igraph as ig
import numpy as np
import torch
from models.gnn_model import MultiplicativeDiffusionGNN
from diffusion_models.independent_cascade import independent_cascade, optimized_independent_cascade, dmp_ic, dynamic_message_passing, ALE_heuristic, monte_carlo_ic, modified_ALE, simulate_multiplicative_path_survival, ALE_heuristic_transpose, IC_approx, IC_approx_vectorized

def generate_y(G, edge_probs, x, w_xy, b_xy, t, w_ty, b_ty, b, num_sim=1000, damp_factor=0.1):
    u = 1/(1+np.exp(-np.dot(x,w_xy)))
    print("u: ", u.mean())
    u_N = make_uN(u, G, edge_probs, num_sim=num_sim, damp_factor=damp_factor)
    print("b: ", b)
    print("b_xy*u_N: ", b_xy*u_N.mean())
    print("(b_ty/(1+np.exp(-np.dot(x,w_ty)))): ", ((b_ty/(1+np.exp(-np.dot(x,w_ty))))).mean())
    value = b + b_xy*u_N+(b_ty/(1+np.exp(-np.dot(x,w_ty))))*np.array(t)
    return 1/(1+np.exp(-value))

def generate_contagion_outcome(G, edge_probs, y, diffusion_method="IC", num_sim=1000, **kwargs):
    if diffusion_method=="IC":
        dict_y = optimized_independent_cascade(G, y, edge_probs, k=num_sim)
        y_prime = np.array([dict_y[id] for id in dict_y])
    elif diffusion_method=="model":
        model = kwargs.get('model', None)
        data = kwargs.get('data', None)
        model.eval()
        with torch.no_grad():
            data.x=y
            y_prime = model(data.x, data.edge_index, data.edge_attr)
    elif diffusion_method=="ALE":
        num_steps = kwargs.get('num_steps', 4)
        y_prime=ALE_heuristic(G, edge_probs, y, num_steps)
    elif diffusion_method=="ModifiedALE":
        num_steps = kwargs.get('num_steps', 4)
        y_prime=modified_ALE(G, y, edge_probs, num_steps)
    elif diffusion_method=="IC_approx":
        num_steps = kwargs.get('num_steps', 10)
        y_prime=IC_approx_vectorized(G, edge_probs, y, num_steps, 1)
    else:
        print("Method not viable!")
        return -1
    return y_prime

def make_uN(u, G, edge_probs, num_sim=1000, damp_factor=0.1):
    damp_dict = {}
    for edge in edge_probs:
        damp_dict[edge] = edge_probs[edge]*damp_factor
    dict_u = optimized_independent_cascade(G, u, damp_dict, k=num_sim)
    u_N = np.array([dict_u[id] for id in dict_u])
    return u_N