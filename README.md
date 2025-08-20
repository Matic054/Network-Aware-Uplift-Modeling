# INVERSE INFECTION GNN

Code used in masters thesis: Network-Aware Uplift Modeling through Influence
Maximization Techniques. 
## CONTRIBUTIONS
The project has 3 main contributions:
1. IC_approx is an efficient and differentiable function to approximate the independent cascade
2. ICApproxLearnableModule is a nn module to that learns edge probabilities based on observed diffusion patterns
3. A network-aware uplift modeling approach is implemented and tested
## PROJECT STRUCTURE
The python code folders include:
- graph_generation:
  - generate_graph.py for generating synthetic graph data
  - generate_outcome.py for generating outcomes on top of graph data that is relavant to uplif modeling
- diffusion_models:
  - edge_probabilities.py for generating the edge probabilities used in the independent cascade model
  - independent_cascade.py functions to approximate the independent cascade model (Monte Carlo, IC_Approx, ALE, ...)
- models:
  - gnn_model.py holds nn modules to approximate the independent cascade, and modules for edge probability predictions
- training:
  - train_edge_gnn.py functions for training and running nn modules
- influence_maximization:
  - im.py holds functions for treatmnet allocation used in uplit modeling
- plotting_tools:
  - prob_plots.py functions to plots a posteriori and edge probabilities
## DATA AND DEPENDENCIES
The project includes 3 notebooks that hold the experimental evaluations of the above-stated contributions. The experiments were done on synthetic and semi-synthetic data-sets. The semi-synthetic data-sets were obtained from: 
R. A. Rossi in N. K. Ahmed, The Network Data Repository with Interactive
Graph Analytics and Visualization. Proc. AAAI Conf. Artif. Intell. (2015) 

Some of the dependencies include:
- matplotlib==3.8.4
- networkx==3.2.1
- numpy==2.3.2
- python_igraph==0.11.8
- scikit_learn==1.4.2
- scipy==1.16.1
- torch==2.5.1
- torch_geometric==2.6.1
- torch_scatter==2.1.2+pt25cu124
