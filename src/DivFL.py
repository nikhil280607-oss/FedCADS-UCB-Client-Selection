import torch
import torch.nn as nn
import copy
import models
import numpy as np

def client_update(client_model, dataloader, local_epochs=3, lr=0.01, is_probe=False):
    # Step 1: Store initial CPU weights
    initial_weights = copy.deepcopy(client_model.state_dict())

    # Step 2: Move model to device and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model = client_model.to(device)
    client_model.train()
    
    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)

    if isinstance(client_model, models.SVM):
        criterion = nn.MultiMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    epochs_to_run = 1 if is_probe else local_epochs

    for epoch in range(epochs_to_run):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if is_probe:
                break

    # Step 3: Move model back to CPU
    client_model = client_model.to('cpu') 
    delta = {}
    new_weights = client_model.state_dict() 
    
    # Step 4: Compute and return weight deltas
    for k in new_weights.keys():
        delta[k] = new_weights[k] - initial_weights[k]

    return delta, new_weights

def flatten_weights(weights_dict):
    return torch.cat([v.flatten() for v in weights_dict.values()])

def divfl_client_selection(all_deltas, num_to_select=10):
    num_clients = len(all_deltas)
    if num_to_select >= num_clients:
        return list(range(num_clients))

    # Step 5: Flatten weights for distance calculation
    flat_deltas = [flatten_weights(delta) for delta in all_deltas]
    dist_matrix = np.zeros((num_clients, num_clients))
    
    # Step 6: Calculate pairwise distances between client updates
    for i in range(num_clients):
        for j in range(num_clients):
            dist_matrix[i, j] = torch.norm(flat_deltas[i] - flat_deltas[j]).item()

    selected_clients = []
    unselected_clients = list(range(num_clients))
    min_distances = np.full(num_clients, np.inf)

    # Step 7: Select clients greedily to maximize diversity
    for step in range(num_to_select):
        best_client = -1
        best_marginal_gain = -np.inf

        for i in unselected_clients:
            new_distances = np.minimum(min_distances, dist_matrix[:, i])
            current_cost = np.sum(min_distances)
            new_cost = np.sum(new_distances)
            gain = current_cost - new_cost if current_cost != np.inf else -new_cost

            if gain > best_marginal_gain:
                best_marginal_gain = gain
                best_client = i

        selected_clients.append(best_client)
        unselected_clients.remove(best_client)
        min_distances = np.minimum(min_distances, dist_matrix[:, best_client])

    return selected_clients