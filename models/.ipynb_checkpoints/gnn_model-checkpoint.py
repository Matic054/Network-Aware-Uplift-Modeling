import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import NNConv, global_mean_pool, Sequential, Linear
from torch_scatter import scatter_add
from torch_scatter import scatter_mul
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

class GNN(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=8, num_hidden=0, dropout=0):
        super(GNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_hidden):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, out_channels))

        self.final_mlp = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, edge_probs):
        for layer in self.gcn_layers:
            x = F.relu(layer(x, edge_index, edge_weight=edge_probs))
            x = self.dropout(x)
        x = self.final_mlp(x)
        x = self.dropout(x)
        x = (x-x.min())/(x.max()-x.min())
        return x#torch.sigmoid(x)

class simpleEdgePrediction(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels,dropout_p=0.2):
        super().__init__()
        self.num_par = node_in_channels+edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.num_par,2*self.num_par),
            nn.Dropout(dropout_p),
            nn.Linear(2*self.num_par, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]
        x_combined = x_src + x_dst
        logits = self.edge_mlp(torch.cat([x_combined, edge_attr], dim=-1)).squeeze()
        return torch.sigmoid(logits)

class EdgeGNN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels=32):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src = self.node_encoder(x[src])
        x_dst = self.node_encoder(x[dst])
        combined = torch.cat([x_src, x_dst, edge_attr], dim=1)
        edge_logits = self.edge_mlp(combined).squeeze()
        return torch.sigmoid(edge_logits)  # [num_edges]

class LearnableDiffusionLayer(nn.Module):
    def __init__(self, hidden_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.self_loop_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, edge_probs):
        src, dst = edge_index
        messages = x[src] * edge_probs.view(-1, 1) * self.weight
        out = scatter_add(messages, dst, dim=0, dim_size=x.size(0))
        out += x * self.self_loop_weight
        return torch.clamp(x + out, 0, 1)

class LearnableDiffusionGNN(nn.Module):
    def __init__(self, num_steps=3, hidden_dim=1):
        super().__init__()
        self.decay_coeff = nn.Parameter(torch.tensor(0.1)) 
        self.layers = nn.ModuleList([LearnableDiffusionLayer(hidden_dim) for _ in range(num_steps)])

    def forward(self, x, edge_index, edge_probs):
        out = x
        for idx, layer in enumerate(self.layers):  
            update = layer(out, edge_index, edge_probs)
            out = out + (update - out) * torch.exp(-idx * self.decay_coeff)  
        return out

class ALE(nn.Module):
    '''
    GNN for estimnating the independent cascade based on ALE model from:
    https://doi.org/10.14232/actacyb.21.1.2013.4
    '''
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.weights = nn.Parameter(torch.ones(num_steps))  # w_1 to w_k

    def forward(self, x, edge_index, edge_probs):
        """
        x: (num_nodes, 1) initial infection probabilities
        edge_index: (2, num_edges) where edge_index[0] = source, edge_index[1] = target
        edge_probs: (num_edges,) edge infection probabilities
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        num_edges = edge_probs.size(0)

        current_x = x.clone()
        result = self.weights[0] * current_x  # w1 * I * x

        for k in range(1, self.num_steps):
            # Map x to edges via source (incidence matrix A)
            edge_vals = x[src] * edge_probs.view(-1, 1)  # shape: (num_edges, 1)

            # Scatter to nodes via edge destinations (A^T)
            current_x = scatter_add(edge_vals, dst, dim=0, dim_size=num_nodes)

            # Accumulate weighted step
            result += self.weights[k] * current_x  # w_k * (A^T)^k x

            # Prepare for next step
            x = current_x

        return result

class ModifiedALE(nn.Module):
    '''
    A GNN model that improves on ALE from:
    https://doi.org/10.14232/actacyb.21.1.2013.4
    '''
    def __init__(self, num_steps, num_nodes, num_edges, dropout=0.3):
        super().__init__()
        self.num_steps = num_steps
        self.time_decay = nn.Parameter(torch.linspace(0, 1, self.num_steps))
        self.node_bias = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(p=dropout)
        self.edge_weight = nn.Parameter(torch.ones(self.num_steps))

    def forward(self, x, edge_index, edge_probs):
        """
        x: (num_nodes, 1) initial infection probabilities
        edge_index: (2, num_edges)
        edge_probs: (num_edges,) edge infection probabilities
        """
        src, dst = edge_index
        num_nodes = x.size(0)

        survival_prob = 1 - x
        current_x = x.clone()

        for i in range(self.num_steps):
            messages = current_x[src] * edge_probs.view(-1, 1) * self.edge_weight[i]
            messages *= torch.exp(-self.time_decay[i]**2)
            current_x = torch.zeros_like(x).scatter_add_(0, dst.view(-1, 1).expand(-1, 1), messages)
            current_x = current_x + self.node_bias

            # Survival update (multiplicative)
            survival_prob *= (1 - current_x) 

        final_infection = 1 - survival_prob
        return torch.clamp(final_infection,0,1)

class MultiplicativeDiffusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_weight = nn.Parameter(torch.ones(1))  
        self.self_loop = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, edge_probs):
        """
        x: (num_nodes, 1) node infection probabilities
        edge_index: (2, num_edges)
        edge_probs: (num_edges,)
        """
        src, dst = edge_index

        # Compute 1 - infection attempt
        messages = 1 - (x[src] * edge_probs.view(-1, 1) * self.edge_weight)  # (num_edges, 1)

        # Aggregate incoming messages multiplicatively
        agg_messages = scatter_mul(messages, dst, dim=0, dim_size=x.size(0)) # (num_nodes, 1)

        return agg_messages + self.self_loop*x

class MultiplicativeDiffusionGNN(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        self.bais = nn.Parameter(torch.tensor(0.01))
        self.layers = nn.ModuleList([
            MultiplicativeDiffusionLayer() for _ in range(num_steps)
        ])

    def forward(self, x, edge_index, edge_probs):
        """
        x: (num_nodes, 1) initial infection probs (prior)
        """
        agg_messages_total = torch.ones_like(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_probs)  # (num_nodes, 1)
            agg_messages_total = agg_messages_total * x  

        # Final infection probability
        out = 1 - agg_messages_total

        return torch.sigmoid(out)


class AdditiveDiffusionGNN(nn.Module):
    '''
    GNN for estimnating the independent cascade based on:
    https://arxiv.org/abs/2108.04623
    '''
    def __init__(self, in_dim, hidden1, hidden2, dropout):
        super().__init__()

        # Two message-passing layers based on adjacency + node features
        self.fc1 = nn.Linear(2 * in_dim, hidden1)
        self.fc2 = nn.Linear(2 * hidden1, hidden2)
        self.fc_out = nn.Linear(in_dim + hidden1 + hidden2, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)

    def forward(self, x, edge_index, edge_probs):
        """
        x: (num_nodes, 1) initial infection probabilities
        edge_index: (2, num_edges)
        edge_probs: (num_edges,) edge probabilities (learned or static)
        """

        src, dst = edge_index
        lst = [x]

        # 1st message passing: weighted neighbor aggregation
        msg1 = edge_probs.view(-1, 1) * x[src]             # (num_edges, 1)
        agg1 = scatter_add(msg1, dst, dim=0, dim_size=x.size(0))  # (num_nodes, 1)
        x1 = torch.cat([x, agg1], dim=1)                   # (num_nodes, 2)
        x1 = self.relu(self.fc1(x1))
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        lst.append(x1)

        # 2nd message passing: again with transformed x1
        msg2 = edge_probs.view(-1, 1) * x1[src]
        agg2 = scatter_add(msg2, dst, dim=0, dim_size=x.size(0))
        x2 = torch.cat([x1, agg2], dim=1)
        x2 = self.relu(self.fc2(x2))
        x2 = self.bn2(x2)
        x2 = self.dropout(x2)
        lst.append(x2)

        # Skip connections and final prediction
        out = torch.cat(lst, dim=1)
        out = torch.sigmoid(self.fc_out(out))  # Probabilities in (0, 1)

        return out  # Final infection probabilities

from torch_geometric.utils import softmax
from torch_geometric.utils import add_self_loops

class ScalarGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_mlp=None, aggr='mean'):
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = edge_mlp or nn.Linear(1, 1, bias=False)  # 1 -> 1

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1.0)

        # Linear transform node features first
        x_transformed = self.lin(x)

        return self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_weight = torch.sigmoid(self.edge_mlp(edge_attr))  # [num_edges, 1]
        return edge_weight * x_j  # scale message by edge weight

class DeepECCNet(nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels, num_layers, dropout=0.3):
        super(DeepECCNet, self).__init__()

        assert in_channels_edge == 1, "Only 1 edge feature (edge probability)."
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        # GNN input: exclude treatment feature (assumed to be x[:, 0])
        gnn_in_channels = in_channels_node - 1

        # First layer
        '''nn0 = nn.Linear(in_channels_edge, gnn_in_channels * hidden_channels, bias=False)
        self.edge_mlps.append(nn0)
        self.convs.append(NNConv(gnn_in_channels, hidden_channels, nn0, aggr='mean'))'''
        edge_mlp0 = nn.Linear(1, 1, bias=False)
        self.edge_mlps.append(edge_mlp0)
        self.convs.append(ScalarGNNConv(gnn_in_channels, hidden_channels, edge_mlp0))

        # Hidden GNN layers
        for _ in range(num_layers - 1):
            '''nn_k = nn.Linear(in_channels_edge, hidden_channels * hidden_channels, bias=False)
            self.edge_mlps.append(nn_k)
            self.convs.append(NNConv(hidden_channels, hidden_channels, nn_k, aggr='mean'))'''
            edge_mlp = nn.Linear(1, 1, bias=False)
            self.edge_mlps.append(edge_mlp)
            self.convs.append(ScalarGNNConv(hidden_channels, hidden_channels, edge_mlp))

        # Final GNN → MLP classifier head
        self.lin1 = nn.Linear(hidden_channels + 1, hidden_channels // 2)  # +1 for treatment effect
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

        # Treatment effect MLP: takes full input (with treatment)
        self.t_lin1 = nn.Linear(in_channels_node, in_channels_node // 2)
        self.t_lin2 = nn.Linear(in_channels_node // 2, 1)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # Save full x with treatment for treatment prediction
        x_full = x  # [num_nodes, in_channels_node]

        # 1. Treatment effect prediction
        t = F.relu(self.t_lin1(x_full))
        t = F.dropout(t, p=self.dropout, training=self.training)
        t = torch.sigmoid(self.t_lin2(t))  # shape [num_nodes, 1]

        # 2. GNN on features excluding treatment assignment
        x = x[:, 1:]  # remove treatment feature
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Pool GNN output if batch exists
        if batch is not None:
            x = global_mean_pool(x, batch)
            t = global_mean_pool(t, batch)

        # 4. Combine GNN and treatment output
        x = torch.cat([x, t], dim=1)  # shape [batch_size, hidden_channels + 1]

        # 5. Final MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.lin2(x)).squeeze(-1)  # shape [batch_size]

        return x

class ICApproxLayer(nn.Module):
    def __init__(self, num_steps, num_nodes):
        super().__init__()
        self.num_steps = num_steps
        self.num_nodes = num_nodes

    def forward(self, prior_probs, edge_index, edge_probs):
        src = edge_index[0]  # Corrected
        dst = edge_index[1]  # Corrected

        # Construct A[v, u] = p_{uv}
        indices = torch.stack([dst, src], dim=0)
        values = edge_probs
        A = torch.sparse_coo_tensor(indices, values, (self.num_nodes, self.num_nodes), device=prior_probs.device)
        A = A.coalesce()

        P = torch.zeros((self.num_steps + 1, self.num_nodes), dtype=prior_probs.dtype, device=prior_probs.device)
        P[0] = prior_probs

        for t in range(1, self.num_steps + 1):
            product_term = torch.prod(1 - P[:t], dim=0)
            delta = torch.sparse.mm(A, P[t - 1].unsqueeze(1)).squeeze(1)
            influence_term = torch.exp(-delta)
            P[t] = product_term * (1 - influence_term)

        final_probs = 1 - torch.prod(1 - P, dim=0)
        return final_probs.view(-1, 1)

class InfluenceSpreadNN(nn.Module):
    def __init__(self, num_steps):
        super(InfluenceSpreadNN, self).__init__()
        self.k = num_steps  # Number of propagation steps

    def forward(self, prior_probs, edge_index, edge_probs):
        num_nodes = prior_probs.size(0)
        p_t_u = prior_probs.clone()  # p_0(u)
        prod_terms = torch.ones_like(prior_probs)  # \prod_{t=0}^k (1 - p_t(u))

        for t in range(self.k):
            # For t > 0, compute influence from neighbors
            source, target = edge_index  # v -> u: source = v, target = u
            p_v = p_t_u[source]
            p_vu = edge_probs
            message = p_vu * p_v  # p_{vu} * p_{t-1}(v)
            
            # Aggregate messages to target nodes (sum over incoming edges)
            agg = scatter_add(message, target, dim=0, dim_size=num_nodes)

            # Compute (1 - exp(-agg))
            delta = 1 - torch.exp(-agg)

            # (prod_{i=0}^{t-1}(1 - p_i(u))) already tracked by prod_terms
            new_p_t_u = prod_terms * delta  # p_t(u)

            # Update prod_terms *= (1 - p_t(u))
            prod_terms *= (1 - new_p_t_u)

            # Prepare for next iteration
            p_t_u = new_p_t_u

        # Final prediction: p'_u = 1 - \prod_{t=0}^k (1 - p_t(u)) = 1 - prod_terms
        p_final = 1 - prod_terms + prior_probs
        return p_final.view(-1, 1)

class ICApproxLossModule(nn.Module):
    def __init__(self):
        super(ICApproxLossModule, self).__init__()

    def forward(self, G, prior_probs, edge_index, edge_probs, k):
        n = G.number_of_nodes()
        node_list = list(G.nodes())
        node_index = {u: i for i, u in enumerate(node_list)}

        # Convert edge_index to graph-relative node indices
        src = edge_index[0]
        dst = edge_index[1]

        # Map tensor node IDs to graph node indices (if needed)
        src = torch.tensor([node_index[int(u.item())] for u in src], device=prior_probs.device)
        dst = torch.tensor([node_index[int(v.item())] for v in dst], device=prior_probs.device)

        # Build sparse adjacency A[v, u] = p_{uv}
        indices = torch.stack([dst, src], dim=0)
        values = edge_probs
        values.requires_grad_(True)
        A = torch.sparse_coo_tensor(indices, values, (n, n), device=prior_probs.device)
        A = A.coalesce()

        # Diffusion
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

class ICApproxLearnableModule(nn.Module):
    def __init__(self, k, edge_index, init_edge_probs=None, max_edge=1):
        super(ICApproxLearnableModule, self).__init__()
        self.max_edge = max_edge
        self.k = k
        self.edge_index = edge_index  # shape: [2, num_edges]
        self.num_edges = edge_index.size(1)

        if init_edge_probs is None:
            init_edge_probs = torch.ones(self.num_edges)*0.1  # small default init

        #self.edge_probs = nn.Parameter(init_edge_probs)  # now a learnable parameter
        self.raw_edge_logits = nn.Parameter(init_edge_probs)

    def forward(self, G, prior_probs):
        edge_probs = torch.sigmoid(self.raw_edge_logits)#.clip(0,self.max_edge)
        n = G.number_of_nodes()
        device = prior_probs.device

        # Create node-to-index mapping
        node_list = list(G.nodes())
        node_index = {u: i for i, u in enumerate(node_list)}

        # Map edge indices to correct tensor indices
        src = self.edge_index[0]
        dst = self.edge_index[1]
        src = torch.tensor([node_index[int(u.item())] for u in src], device=device)
        dst = torch.tensor([node_index[int(v.item())] for v in dst], device=device)

        # Build sparse adjacency matrix A[v, u] = p_{uv}
        indices = torch.stack([dst, src], dim=0)  # Note: dst as rows, src as cols
        A = torch.sparse_coo_tensor(indices, edge_probs, (n, n), device=device).coalesce()

        # Influence propagation
        prior_probs = prior_probs.view(-1)
        P_list = [prior_probs]

        for t in range(1, self.k + 1):
            stacked = torch.stack(P_list, dim=0)  # shape [t, n]
            product_term = torch.prod(1 - stacked, dim=0)

            delta = torch.sparse.mm(A, P_list[-1].view(-1, 1)).squeeze(1)
            influence_term = torch.exp(-delta.clamp(min=0, max=10))  # stability
            P_next = product_term * (1 - influence_term)

            P_list.append(P_next.view(-1))  # keep 1D

        P_tensor = torch.stack(P_list, dim=0)  # shape [k+1, n]
        final_probs = 1 - torch.prod(1 - P_tensor, dim=0)
        return final_probs, edge_probs