import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data

class GeneticEdgeOptimizer:
    def __init__(self, f, input_dim, population_size=50, mutation_rate=0.1, crossover_rate=0.5, device='cpu'):
        self.f = f
        self.input_dim = input_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.device = device
        self.population = [torch.randn(input_dim, device=device, requires_grad=False) for _ in range(population_size)]

    def predict(self, weights, edge_inputs):
        logits = torch.matmul(edge_inputs, weights)
        return torch.sigmoid(logits)

    def evaluate(self, weights, edge_inputs, true_edge_weights):
        preds = self.predict(weights, edge_inputs)
        loss = F.mse_loss(self.f(preds), self.f(true_edge_weights))
        return loss.item()

    def select(self, scores):
        ranked = sorted(zip(scores, self.population), key=lambda x: x[0])
        self.population = [w for _, w in ranked[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        mask = torch.rand(self.input_dim, device=self.device) < self.crossover_rate
        child = parent1.clone()
        child[mask] = parent2[mask]
        return child

    def mutate(self, weights):
        mutation = torch.randn_like(weights) * self.mutation_rate
        return weights + mutation

    def step(self, edge_inputs, true_edge_weights):
        scores = [self.evaluate(w, edge_inputs, true_edge_weights) for w in self.population]
        self.select(scores)
        new_population = self.population.copy()
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(self.population, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population
        best_score = min(scores)
        best_weights = self.population[scores.index(best_score)]
        return best_weights, best_score

class ICApproxLossModule(torch.nn.Module):
    def __init__(self):
        super(ICApproxLossModule, self).__init__()

    def forward(self, G, prior_probs, edge_index, edge_probs, k):
        n = G.number_of_nodes()
        node_list = list(G.nodes())
        node_index = {u: i for i, u in enumerate(node_list)}
        src = edge_index[0]
        dst = edge_index[1]
        src = torch.tensor([node_index[int(u.item())] for u in src], device=prior_probs.device)
        dst = torch.tensor([node_index[int(v.item())] for v in dst], device=prior_probs.device)
        indices = torch.stack([dst, src], dim=0)
        values = edge_probs
        values.requires_grad_(True)
        A = torch.sparse_coo_tensor(indices, values, (n, n), device=prior_probs.device).coalesce()
        P_list = [prior_probs]
        for t in range(1, k + 1):
            stacked = torch.stack(P_list, dim=0)
            product_term = torch.prod(1 - stacked, dim=0)
            delta = torch.sparse.mm(A, P_list[-1].unsqueeze(1)).squeeze(1)
            influence_term = torch.exp(-delta)
            P_next = product_term * (1 - influence_term)
            P_list.append(P_next)
        P_tensor = torch.stack(P_list, dim=0)
        final_probs = 1 - torch.prod(1 - P_tensor, dim=0)
        return final_probs

def get_edge_input_matrix(x, edge_index, edge_attr):
    src, dst = edge_index
    x_src = x[src]
    x_dst = x[dst]
    return torch.cat([x_src, x_dst, edge_attr], dim=-1)

def run_ga_training(G, data, edge_dict, true_posterior_tensor, generations=100):
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x
    edge_inputs = get_edge_input_matrix(x, edge_index, edge_attr)
    true_edge_probs = torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32)
    input_dim = edge_inputs.shape[1]

    prior_probs = x[:, 0]  # <- make sure this is [num_nodes], not [num_edges]

    diff_model = ICApproxLossModule()
    f = lambda edge_probs: diff_model(G, prior_probs, edge_index, edge_probs, k=10)

    ga = GeneticEdgeOptimizer(f=f, input_dim=input_dim, population_size=50, mutation_rate=0.05, crossover_rate=0.1)
    best_w = None

    for gen in range(generations):
        best_w, best_loss = ga.step(edge_inputs, true_edge_probs)
        if gen % 10 == 0:
            print(f"Generation {gen}, Loss: {best_loss:.4f}")

    return best_w

def evaluate_on_test_graph(G_t, data_t, best_w, true_posterior_tensor_t):
    edge_inputs_t = get_edge_input_matrix(data_t.x, data_t.edge_index, data_t.edge_attr)
    predicted_edge_probs_t = torch.sigmoid(edge_inputs_t @ best_w)
    predicted_posterior_t = ICApproxLossModule()(G_t, data_t.x[:, 0], data_t.edge_index, predicted_edge_probs_t, k=10).unsqueeze(1)
    rmse_t = torch.sqrt(F.mse_loss(predicted_posterior_t, true_posterior_tensor_t))
    print(f"Test RMSE (GA model): {rmse_t.item():.4f}")