import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import from_networkx
from models.gnn_model import LearnableDiffusionGNN, MultiplicativeDiffusionGNN, AdditiveDiffusionGNN, ALE, ModifiedALE, DeepECCNet, ICApproxLayer, GNN, InfluenceSpreadNN, ICApproxLossModule, ICApproxLearnableModule
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):  # eps to avoid sqrt(0)
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

def pearson_corr(x, y):
    corr = torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + 1e-8)
    return corr

def combined_loss(pred, target, alpha=0.5):
    mse = F.mse_loss(pred, target)
    corr = pearson_corr(pred, target)
    # We subtract correlation so that higher correlation = lower loss
    loss = alpha * mse + (1 - alpha) * (1 - corr)
    return loss

def combined_loss_bce_mse(pred, target, alpha1=0.333, alpha2=0.333):
    mse = F.mse_loss(pred, target)
    corr = pearson_corr(pred, target)
    bce = torch.nn.BCELoss()(pred, target)
    # We subtract correlation so that higher correlation = lower loss
    loss = alpha1 * mse + alpha2 * (1 - corr) + (1-alpha1-alpha2)*bce
    return loss

def train_diffusion_gnn(data, independent_cascade, num_steps=5, epochs=100, lr=0.01,hidden_dim=1,use_difference=False, model_to_use="multiplicative", alpha=0.7, alpha1=0.333, alpha2=0.333, num_sim=1000, print_loss=True, BCE=True, dropout=0.3, reg_const=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_to_use=="multiplicative":
        model = MultiplicativeDiffusionGNN(num_steps=num_steps).to(device)
    elif model_to_use=="learnableDiffusion":
        model = LearnableDiffusionGNN(num_steps=num_steps, hidden_dim=hidden_dim).to(device)
    elif model_to_use=="additive":
        model = AdditiveDiffusionGNN(
            in_dim=1,
            hidden1=32,
            hidden2=16,
            dropout=0.2
        ).to(device)
    elif model_to_use=="ALE":
        model = ALE(num_steps=num_steps).to(device)
    elif model_to_use=="ModifiedALE":
        model = ModifiedALE(num_steps=num_steps, num_nodes=data.x.shape[0], num_edges=data.edge_index.shape[1], dropout=dropout).to(device)
    elif model_to_use=="ICApproxLayer":
        model = ICApproxLayer(num_steps=num_steps,num_nodes=data.x.size()[0]).to(device)
    elif model_to_use=="GCN":
        model = GNN(in_channels=1,num_hidden=max(num_steps-2,0), dropout=dropout).to(device)
    elif model_to_use=="InfluenceSpreadNN":
        model = InfluenceSpreadNN(num_steps=num_steps).to(device)
    elif model_to_use=="ICApproxLearnableModule":
        model = ICApproxLearnableModule(k=num_steps, edge_index=data.edge_index).to(device)
    else:
        print("no vaible model")
        return -1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    x = data.x.to(device)  # prior infection probabilities [N, 1]
    edge_index = data.edge_index.to(device)
    edge_probs = data.edge_attr.view(-1).to(device)  # [E]

    # Run independent cascade to get true posteriors
    G = to_networkx(data, to_undirected=False)
    prior_probs = x
    edge_probs_dict = {(int(u), int(v)): edge_probs[i].item() for i, (u, v) in enumerate(edge_index.t())}
    true_posteriors = independent_cascade(G, prior_probs, edge_probs_dict, num_sim)

    # Convert true posteriors to tensor
    true_tensor = torch.tensor(
        [true_posteriors[i] for i in range(x.size(0))], dtype=torch.float32, device=device
    ).unsqueeze(1)

    # Training loop
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if model_to_use == "ICApproxLearnableModule":
            #pred_posterior = model(G, x, edge_index, edge_probs)
            pred_posterior,edge_probs = model(G, x)
        else:
            pred_posterior = model(x, edge_index, edge_probs)
        if use_difference:
            loss = combined_loss(pred_posterior-prior_probs.unsqueeze(1), true_tensor-prior_probs.unsqueeze(1), alpha=alpha)
            #loss = F.mse_loss(pred_posterior-prior_probs.unsqueeze(1), true_tensor-prior_probs.unsqueeze(1))
        else:
            if BCE:
                #criterion = torch.nn.BCELoss()
                #loss = criterion(pred_posterior, true_tensor)
                loss=combined_loss_bce_mse(pred_posterior, true_tensor, alpha1, alpha2)
            else:
                #reg_loss = 0.1*torch.mean(edge_probs * (1 - edge_probs))  # penalize midrange probs
                reg_loss = torch.distributions.kl.kl_divergence(
                        torch.distributions.Normal(edge_probs, torch.ones_like(edge_probs)),
                        torch.distributions.Normal(0.15, 0.5)
                    ).mean()
                loss = combined_loss(pred_posterior.unsqueeze(1), true_tensor, alpha=alpha)+reg_const*reg_loss#+reg_loss
                if loss < 0.0001:
                    break
            #loss = F.mse_loss(pred_posterior, true_tensor)
        loss = loss.to(dtype=torch.float32)

        loss.backward()
        optimizer.step()
        
        if (epoch % 10 == 0 or epoch == epochs - 1) and print_loss:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

def edge_probs_prediction(data, independent_cascade, num_steps=5, epochs=100, lr=0.01, alpha=0.7, num_sim=1000, print_loss=True, reg_const=0.1, initial_edges = None, decay=0.9, max_edge=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if initial_edges == None:
        model = ICApproxLearnableModule(k=num_steps, edge_index=data.edge_index, max_edge=max_edge).to(device)
    else:
        model = ICApproxLearnableModule(k=num_steps, edge_index=data.edge_index, init_edge_probs=initial_edges, max_edge=max_edge).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    x = data.x.to(device)  # prior infection probabilities [N, 1]
    edge_index = data.edge_index.to(device)
    edge_probs = data.edge_attr.view(-1).to(device)  # [E]

    # Run independent cascade to get true posteriors
    G = to_networkx(data, to_undirected=False)
    prior_probs = x
    edge_probs_dict = {(int(u), int(v)): edge_probs[i].item() for i, (u, v) in enumerate(edge_index.t())}
    true_posteriors = independent_cascade(G, prior_probs, edge_probs_dict, num_sim)

    # Convert true posteriors to tensor
    true_tensor = torch.tensor(
        [true_posteriors[i] for i in range(x.size(0))], dtype=torch.float32, device=device
    ).unsqueeze(1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_posterior, edge_probs = model(G, x)
        reg_loss = torch.distributions.kl.kl_divergence(
                        torch.distributions.Normal(edge_probs, torch.ones_like(edge_probs)),
                        torch.distributions.Normal(0.15, 0.5)
                    ).mean()
        loss = combined_loss(pred_posterior.unsqueeze(1), true_tensor, alpha=alpha)+reg_const*reg_loss
        if loss < 0.0001:
            break
        loss = loss.to(dtype=torch.float32)

        loss.backward()
        optimizer.step()
        
        lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if (epoch % 10 == 0 or epoch == epochs - 1) and print_loss:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        pred_posterior, edge_probs = model(G, x)

    return model

def train_edge_model_with_diffusion(edge_gnn, diffusion_gnn, data, true_posterior_tensor, epochs=100, lr=0.01, alpha=1):
    device = data.x.device
    optimizer = torch.optim.Adam(edge_gnn.parameters(), lr=lr)
    
    diffusion_gnn.eval()
    for param in diffusion_gnn.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        edge_gnn.train()
        optimizer.zero_grad()
        
        # Predict edge probabilities
        edge_probs = edge_gnn(data.x, data.edge_index, data.edge_attr)
        edge_probs = edge_probs.to(dtype=torch.float32)

        # Run differentiable diffusion
        prior = data.x[:, 0].unsqueeze(1)  # assuming last column is prior infection
        prior = prior.to(dtype=torch.float32)
        predicted_posterior = diffusion_gnn(prior, data.edge_index, edge_probs)
        predicted_posterior=predicted_posterior.to(dtype=torch.float32)
        #print(predicted_posterior.grad_fn)  # Should NOT be None
        #print("Edge probs grad_fn:", edge_probs.grad_fn)

        # Loss
        #loss = F.mse_loss(predicted_posterior, true_posterior_tensor)
        loss = combined_loss(predicted_posterior, true_posterior_tensor.to(dtype=torch.float32), alpha=alpha)
        loss = loss.to(dtype=torch.float32)
        #print("Loss requires_grad:", loss.requires_grad)
        #print(torch.autograd.grad(loss, edge_probs, retain_graph=True, allow_unused=True))
        loss.backward()
        #for name, param in edge_gnn.named_parameters():
            #print(f"{name}: grad = {param.grad.abs().sum().item():.6f}")
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    return edge_gnn 

def train_edge_model(data_for_diff, edge_gnn, data, true_posterior_tensor, epochs=100, lr=0.01, alpha=1):
    device = data.x.device
    optimizer = torch.optim.Adam(edge_gnn.parameters(), lr=lr)
    diff_model = ICApproxLossModule()

    for epoch in range(epochs):
        edge_gnn.train()
        optimizer.zero_grad()
        
        # Predict edge probabilities
        edge_probs = edge_gnn(data.x, data.edge_index, data.edge_attr)
        edge_probs = edge_probs.to(dtype=torch.float32)
        
        # Update edge_dict for diffusion input
        data_for_diff["edge_dict"] = edge_index_to_dict(data.edge_index, edge_probs)

        # Prepare prior input
        prior = data.x[:, 0].to(dtype=torch.float32)

        predicted_posterior = diff_model(
            data_for_diff["G"],
            prior,
            data.edge_index,
            edge_probs,
            data_for_diff["k"]
        ).unsqueeze(1)

        # Compute loss
        loss = combined_loss(predicted_posterior, true_posterior_tensor.to(dtype=torch.float32), alpha=alpha)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return edge_gnn


def edge_index_to_dict(edge_index, edge_probs):
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    edges = list(zip(src, dst))
    edge_dict = {edge: edge_probs[i] for i, edge in enumerate(edges)}  # use tensor elements
    return edge_dict

def train_DeepECCNet(data,lr=0.01,epochs=300,num_layers=5, loss_type="mse", alpha=0.5, hidden_channels=64, dropout=0.3, w_decay=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepECCNet(in_channels_node=data.x.size()[1], in_channels_edge=data.edge_attr.size()[1], hidden_channels=hidden_channels,num_layers=num_layers,dropout=dropout).to(device)
    if w_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    data = data.to(device)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_y = model(data.x, data.edge_index, data.edge_attr)
        if loss_type == "mse":
            loss = F.mse_loss(pred_y, data.y)
        elif loss_type == "BCE":
            criterion = torch.nn.BCELoss()
            loss = criterion(pred_y, data.y)
        elif loss_type == "alpha":
            loss = combined_loss(pred_y, data.y, alpha=alpha)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model 
