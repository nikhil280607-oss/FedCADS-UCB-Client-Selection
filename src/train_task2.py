import torch
import copy
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt

import data_setup
import models
import main
import shapley

class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, label_map):
        self.original_dataset = original_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        new_label = self.label_map.get(label_val, label_val)
        return image, new_label

def aggregate_models(global_model, selected_deltas):
    global_weights = global_model.state_dict()
    avg_delta = {k: torch.zeros_like(v) for k, v in global_weights.items()}
    for delta in selected_deltas:
        for k in avg_delta.keys():
            avg_delta[k] += delta[k]
    num_selected = len(selected_deltas)
    for k in avg_delta.keys():
        avg_delta[k] = avg_delta[k] / float(num_selected)
        global_weights[k] += avg_delta[k]
    global_model.load_state_dict(global_weights)

def test_global_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

test_loader = DataLoader(data_setup.test_dataset, batch_size=64, shuffle=False)
val_subset = Subset(data_setup.test_dataset, range(500))
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
num_clients = 50
TOTAL_ROUNDS = 100

models_to_test = {
    "Logistic Regression": models.LogisticRegression,
    "SVM": models.SVM,
    "Deep Neural Network": models.DNN,
    "Simple CNN": models.SimpleCNN
}

results = {}

poisoned_client_ids = list(range(15))
label_mapping = {i: (i + 1) % 10 for i in range(10)}

for model_name, model_class in models_to_test.items():
    print(f"\nEVALUATING MODEL: {model_name} (WITH CONCEPT DRIFT)")

    divfl_accuracy_history = []
    sfedavg_accuracy_history = []

    global_model_divfl = model_class()
    current_clients_data = data_setup.create_non_iid_clients(data_setup.train_dataset, num_clients=num_clients, batch_size=16)

    for round_num in range(TOTAL_ROUNDS):
        if round_num == 50:
            for cid in poisoned_client_ids:
                old_loader = current_clients_data[cid]
                poisoned_ds = PoisonedDataset(old_loader.dataset, label_mapping)
                current_clients_data[cid] = DataLoader(poisoned_ds, batch_size=16, shuffle=True)

        candidate_pool = np.random.choice(range(num_clients), size=20, replace=False)
        all_deltas = []
        for client_id in candidate_pool:
            delta, _ = main.client_update(copy.deepcopy(global_model_divfl), current_clients_data[client_id], local_epochs=3)
            all_deltas.append(delta)

        selected_indices = main.divfl_client_selection(all_deltas, num_to_select=10)
        selected_deltas = [all_deltas[i] for i in selected_indices]
        aggregate_models(global_model_divfl, selected_deltas)

        acc = test_global_model(global_model_divfl, test_loader)
        divfl_accuracy_history.append(acc)
        if (round_num + 1) % 10 == 0:
            print(f"  Round {round_num + 1}/{TOTAL_ROUNDS} - Accuracy: {acc * 100:.2f}%")

    global_model_sfedavg = model_class()
    relevance_scores = {i: 0.0 for i in range(num_clients)}
    current_clients_data = data_setup.create_non_iid_clients(data_setup.train_dataset, num_clients=num_clients, batch_size=16)

    for round_num in range(TOTAL_ROUNDS):
        if round_num == 50:
            for cid in poisoned_client_ids:
                old_loader = current_clients_data[cid]
                poisoned_ds = PoisonedDataset(old_loader.dataset, label_mapping)
                current_clients_data[cid] = DataLoader(poisoned_ds, batch_size=16, shuffle=True)

        selected_clients = shapley.s_fedavg_select_clients(relevance_scores, num_to_select=10)
        client_deltas = {}
        for client_id in selected_clients:
            delta, _ = main.client_update(copy.deepcopy(global_model_sfedavg), current_clients_data[client_id], local_epochs=3)
            client_deltas[client_id] = delta

        shapley_vals = shapley.compute_shapley_values(global_model_sfedavg, client_deltas, val_loader, R=3)

        selected_deltas = []
        for cid, s_val in shapley_vals.items():
            relevance_scores[cid] = (0.75 * relevance_scores[cid]) + (0.25 * s_val)
            selected_deltas.append(client_deltas[cid])

        aggregate_models(global_model_sfedavg, selected_deltas)

        acc = test_global_model(global_model_sfedavg, test_loader)
        sfedavg_accuracy_history.append(acc)
        if (round_num + 1) % 10 == 0:
            print(f"  Round {round_num + 1}/{TOTAL_ROUNDS} - Accuracy: {acc * 100:.2f}%")

    results[model_name] = {
        "DivFL": divfl_accuracy_history,
        "S-FedAvg": sfedavg_accuracy_history
    }

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Concept Drift (Targeted Label Noise at Round 50): DivFL vs S-FedAvg', fontsize=16)

axs = axs.flatten()
for idx, (model_name, data) in enumerate(results.items()):
    axs[idx].plot(range(1, TOTAL_ROUNDS + 1), data["DivFL"], label='DivFL', color='blue')
    axs[idx].plot(range(1, TOTAL_ROUNDS + 1), data["S-FedAvg"], label='S-FedAvg', color='orange')
    axs[idx].axvline(x=50, color='red', linestyle='--', label='Poison Injected')
    axs[idx].set_title(model_name)
    axs[idx].set_xlabel('Communication Rounds')
    axs[idx].set_ylabel('Test Accuracy')
    axs[idx].legend()
    axs[idx].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('task2_concept_drift_results.png')
plt.show()