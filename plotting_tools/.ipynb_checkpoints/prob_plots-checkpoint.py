import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
import torch
import numpy as np

def posterior_plots(size_x, size_y, predicted_posterior, true_posterior_tensor, percentile=50):
    plt.figure(figsize=(size_x, size_y))
    plt.hist(predicted_posterior.cpu().numpy(), alpha=0.5, label='Predicted',bins=100)
    plt.hist(true_posterior_tensor.cpu().numpy(), alpha=0.5, label='True',bins=100)
    plt.legend()
    plt.title("Posterior Distributions")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(size_x, size_y))
    plt.hist((predicted_posterior.cpu().numpy()-true_posterior_tensor.cpu().numpy()), bins=100)
    plt.title("Residual Distributions")
    plt.tight_layout()
    plt.show()
    
    # Convert to numpy
    pred_np = predicted_posterior.squeeze().detach().cpu().numpy()
    true_np = true_posterior_tensor.squeeze().detach().cpu().numpy()
    
    # Sort both in descending order
    sorted_pred = np.sort(pred_np)[::-1]
    sorted_true = np.sort(true_np)[::-1]
    
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.plot(sorted_true, label='True Posterior (sorted)', linewidth=2)
    plt.plot(sorted_pred, label='Predicted Posterior (sorted)', linewidth=2, linestyle='--')
    plt.xlabel("Node Rank (sorted by infection prob)")
    plt.ylabel("Infection Probability")
    plt.title("Sorted Node Posterior Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.plot(true_np, label='True Posterior', linewidth=2)
    plt.plot(pred_np, label='Predicted Posterior', linewidth=2, linestyle='--')
    plt.xlabel("Node")
    plt.ylabel("Infection Probability")
    plt.title("Node Posterior Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Convert to numpy
    y_true = true_posterior_tensor.squeeze().detach().cpu().numpy()
    y_score = predicted_posterior.squeeze().detach().cpu().numpy()
    
    # Binarize the true labels
    threshold = np.percentile(y_true, percentile)
    print(f"ROC threshold for {percentile}-percentil is {threshold}")
    y_binary = (y_true >= threshold).astype(int)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    auc_score = roc_auc_score(y_binary, y_score)
    
    # Plot ROC curve
    plt.figure(figsize=(size_x/2, size_y))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Predicted Posterior Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Convert tensors to NumPy arrays
    x = predicted_posterior.squeeze().cpu().numpy()
    y = true_posterior_tensor.squeeze().cpu().numpy()
    
    # Create the scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel('Predicted Posterior')
    plt.ylabel('True Posterior')
    plt.title('Scatter Plot: Predicted vs. True Posterior')
    plt.grid(True)
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='Ideal')  # Optional: y = x line
    plt.legend()
    plt.show()

    pearson = torch.corrcoef(torch.stack([predicted_posterior.squeeze(), true_posterior_tensor.squeeze()]))[0, 1]
    print("The Pearson correlation: ", pearson)
    spearman_corr, _ = spearmanr(predicted_posterior.squeeze().cpu().numpy(),
                              true_posterior_tensor.squeeze().cpu().numpy())
    print("The Spearman correlation: ", spearman_corr)

def edge_plots(size_x, size_y, pred_edge_probs, true_edge_probs):
    plt.figure(figsize=(size_x, size_y))
    plt.hist(pred_edge_probs.cpu().numpy(), alpha=0.5, label='Predicted', bins=100)
    plt.hist(true_edge_probs.cpu().numpy(), alpha=0.5, label='True', bins=100)
    plt.legend()
    plt.title("Posterior Edge Probability Distributions")
    plt.show()
    
    plt.figure(figsize=(size_x, size_y))
    plt.hist((pred_edge_probs.cpu().numpy()-true_edge_probs.cpu().numpy()), bins=100)
    plt.title("Residual Edge Probability Distributions")
    plt.show()
    
    # Convert to numpy
    pred_np = pred_edge_probs.squeeze().detach().cpu().numpy()
    true_np = true_edge_probs.squeeze().detach().cpu().numpy()
    
    # Sort both in descending order
    sorted_pred = np.sort(pred_np)[::-1]
    sorted_true = np.sort(true_np)[::-1]
    
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.plot(sorted_true, label='True Edge Probabilitys (sorted)', linewidth=2)
    plt.plot(sorted_pred, label='Predicted Edge Probabilitys (sorted)', linewidth=2, linestyle='--')
    plt.xlabel("Edge Rank (sorted by prob)")
    plt.ylabel("Edge Probability")
    plt.title("Sorted Edge Infection Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Convert to numpy
    pred_np = pred_edge_probs.squeeze().detach().cpu().numpy()
    true_np = true_edge_probs.squeeze().detach().cpu().numpy()
    
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.plot(true_np, label='True Edge Probability', linewidth=2)
    plt.plot(pred_np, label='Predicted Edge Probabilty', linewidth=2, linestyle='--')
    plt.xlabel("Edge")
    plt.ylabel("Edge Infection Probability")
    plt.title("Edge Infection Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    pearson = torch.corrcoef(torch.stack([pred_edge_probs.squeeze(), true_edge_probs.squeeze()]))[0, 1]
    print("The Pearson correlation: ", pearson)
    spearman_corr, _ = spearmanr(pred_edge_probs.squeeze().cpu().numpy(),
                              true_edge_probs.squeeze().cpu().numpy())
    print("The Spearman correlation: ", spearman_corr)