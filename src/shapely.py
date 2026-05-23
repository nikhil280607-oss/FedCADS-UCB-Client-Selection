import torch
import copy
import numpy as np

def evaluate_model(model, dataloader):
    # Step 1: Evaluate model accuracy on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model = model.to('cpu')
    return correct / total

def compute_shapley_values(global_model, client_deltas, val_loader, R=10):
    # Step 2: Compute Shapley values using Monte Carlo approximation
    client_ids = list(client_deltas.keys())
    shapley_values = {cid: 0.0 for cid in client_ids}
    global_weights = global_model.state_dict()

    # Get baseline accuracy before applying client updates
    baseline_val = evaluate_model(global_model, val_loader)

    for _ in range(R):
        perm = client_ids.copy()
        np.random.shuffle(perm)
        current_val = baseline_val

        for i, cid in enumerate(perm):
            coalition = perm[:i + 1]
            avg_delta = {k: torch.zeros_like(v) for k, v in global_weights.items()}
            
            # Calculate average delta for the current coalition
            for c in coalition:
                for k in avg_delta.keys():
                    avg_delta[k] += client_deltas[c][k]
            for k in avg_delta.keys():
                avg_delta[k] = avg_delta[k] / len(coalition)

            # Apply coalition updates and calculate marginal contribution
            temp_model = copy.deepcopy(global_model)
            temp_weights = temp_model.state_dict()
            for k in temp_weights.keys():
                temp_weights[k] += avg_delta[k]
            temp_model.load_state_dict(temp_weights)

            new_val = evaluate_model(temp_model, val_loader)
            marginal_contribution = new_val - current_val
            shapley_values[cid] += marginal_contribution / R
            current_val = new_val

    return shapley_values

def s_fedavg_select_clients(relevance_scores, num_to_select=10):
    # Step 3: Select clients proportionally based on relevance scores
    client_ids = list(relevance_scores.keys())
    scores = np.array([relevance_scores[cid] for cid in client_ids])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    selected_clients = np.random.choice(client_ids, size=num_to_select, replace=False, p=probs)
    return selected_clients.tolist()