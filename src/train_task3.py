import torch
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset

import data_setup
import models
import main
import shapley
from fedcads import FedCADS

# Step 1: Define a custom dataset for poisoned data
class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, label_map):
        self.original_dataset = original_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        return image, self.label_map.get(label_val, label_val)

# Step 2: Aggregate client deltas using FedAvg
def aggregate_models(global_model, deltas):
    if not deltas:
        return
    w = global_model.state_dict()
    avg = {k: torch.zeros_like(v) for k, v in w.items()}
    for d in deltas:
        for k in avg:
            avg[k] += d[k]
    for k in avg:
        avg[k] /= len(deltas)
        w[k]   += avg[k]
    global_model.load_state_dict(w)

# Step 3: Inject target label poisoning to specific clients
def inject_poison(client_data, poisoned_ids, label_map):
    for cid in poisoned_ids:
        ds = PoisonedDataset(client_data[cid].dataset, label_map)
        client_data[cid] = DataLoader(ds, batch_size=16, shuffle=True)

# Step 4: Display training progress and ETA
def progress(algo, rnd, total, acc, times, extra=""):
    pct   = (rnd + 1) / total * 100
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    eta_s = int(sum(times) / len(times) * (total - rnd - 1))
    m, s  = divmod(eta_s, 60)
    print(f"[{algo}] [{bar}] {pct:5.1f}%  Round {rnd+1:03d}/{total}"
          f"  Acc:{acc*100:5.2f}%  ETA:{m}m{s:02d}s{extra}", flush=True)


# Step 5: Global configuration for the training run
NUM_CLIENTS   = 50
TOTAL_ROUNDS  = 100
POISONED_IDS  = list(range(15))
LABEL_MAP     = {i: (i + 1) % 10 for i in range(10)}
K             = 10

test_loader = DataLoader(data_setup.test_dataset, batch_size=64, shuffle=False)
val_loader  = DataLoader(Subset(data_setup.test_dataset, range(500)), batch_size=64)


# Step 6: Run DivFL algorithm: diversity-based selection
def run_divfl():
    print("\n" + "="*55 + "\n  1/3  DivFL\n" + "="*55, flush=True)
    model       = models.SimpleCNN()
    client_data = data_setup.create_non_iid_clients(data_setup.train_dataset, NUM_CLIENTS, batch_size=16)
    history, times = [], []

    for rnd in range(TOTAL_ROUNDS):
        t = time.time()
        if rnd == 50:
            print("  [!] DivFL: drift injected.", flush=True)
            inject_poison(client_data, POISONED_IDS, LABEL_MAP)

        probe_deltas = [
            main.client_update(copy.deepcopy(model), client_data[c], local_epochs=1, is_probe=True)[0]
            for c in range(NUM_CLIENTS)
        ]
        selected = main.divfl_client_selection(probe_deltas, num_to_select=K)

        deltas = [
            main.client_update(copy.deepcopy(model), client_data[c], local_epochs=3)[0]
            for c in selected
        ]
        aggregate_models(model, deltas)

        acc = shapley.evaluate_model(model, test_loader)
        history.append(acc)
        times.append(time.time() - t)
        progress("DivFL  ", rnd, TOTAL_ROUNDS, acc, times,
                 " ⚠ DRIFT" if rnd == 50 else "")

    return history


# Step 7: Run S-FedAvg algorithm: Shapley-weighted EMA selection
def run_sfedavg():
    print("\n" + "="*55 + "\n  2/3  S-FedAvg\n" + "="*55, flush=True)
    model          = models.SimpleCNN()
    client_data    = data_setup.create_non_iid_clients(data_setup.train_dataset, NUM_CLIENTS, batch_size=16)
    ema_scores     = {k: 0.0 for k in range(NUM_CLIENTS)}
    history, times = [], []

    for rnd in range(TOTAL_ROUNDS):
        t = time.time()
        if rnd == 50:
            print("  [!] S-FedAvg: drift injected.", flush=True)
            inject_poison(client_data, POISONED_IDS, LABEL_MAP)

        selected = shapley.s_fedavg_select_clients(ema_scores, num_to_select=K)

        deltas = {
            c: main.client_update(copy.deepcopy(model), client_data[c], local_epochs=3)[0]
            for c in selected
        }

        sv_map = shapley.compute_shapley_values(model, deltas, val_loader, R=5)

        for c in selected:
            ema_scores[c] = 0.75 * ema_scores[c] + 0.25 * sv_map[c]

        aggregate_models(model, list(deltas.values()))

        acc = shapley.evaluate_model(model, test_loader)
        history.append(acc)
        times.append(time.time() - t)
        progress("S-FedAvg", rnd, TOTAL_ROUNDS, acc, times,
                 " ⚠ DRIFT" if rnd == 50 else "")

    return history


# Step 8: Run FedCADS-UCB algorithm: CUSUM-adaptive quarantine
def run_fedcads():
    print("\n" + "="*55 + "\n  3/3  FedCADS-UCB (Proposed)\n" + "="*55, flush=True)
    h_global    = models.SimpleCNN()
    h_drift     = models.SimpleCNN()
    client_data = data_setup.create_non_iid_clients(data_setup.train_dataset, NUM_CLIENTS, batch_size=16)
    server      = FedCADS(NUM_CLIENTS, fairness_rates=[0.01]*NUM_CLIENTS, beta=0.10)
    history, times = [], []
    prev_val    = shapley.evaluate_model(h_global, val_loader)

    for rnd in range(TOTAL_ROUNDS):
        t = time.time()
        if rnd == 50:
            print("  [!] FedCADS: drift injected.", flush=True)
            inject_poison(client_data, POISONED_IDS, LABEL_MAP)

        curr_val            = shapley.evaluate_model(h_global, val_loader)
        gamma, drift_signal = server.update_drift_and_gamma(prev_val - curr_val)
        prev_val            = curr_val

        scores  = server.get_selection_scores()
        active  = [i for i in range(NUM_CLIENTS) if server.Q_k[i] == 0]
        if len(active) > K:
            selected = [i for i in np.argsort(scores)[::-1] if i in active][:K]
        else:
            selected = active

        deltas = {
            c: main.client_update(copy.deepcopy(h_global), client_data[c], local_epochs=3)[0]
            for c in selected
        }

        sv_map  = shapley.compute_shapley_values(h_global, deltas, val_loader, R=5)
        sv_list = [sv_map[c] for c in selected]
        server.update_client_stats(selected, sv_list)

        sv_dict = dict(zip(selected, sv_list))
        aggregate_models(h_global,
                         [deltas[c] for c in selected if sv_dict[c] >= -server.delta_q])
        aggregate_models(h_drift,
                         [deltas[c] for c in selected if sv_dict[c] <  -server.delta_q])

        quarantined = server.get_quarantined_clients()
        if quarantined and drift_signal < 0.1:
            for cid in quarantined[:5]:
                pd, _ = main.client_update(copy.deepcopy(h_global), client_data[cid],
                                           local_epochs=1, is_probe=True)
                pm = copy.deepcopy(h_global)
                pw = pm.state_dict()
                for k in pw:
                    pw[k] += pd[k]
                pm.load_state_dict(pw)
                if shapley.evaluate_model(pm, val_loader) > prev_val + 0.01:
                    server.Q_k[cid]     = 0
                    server.R_tilde[cid] = 0.0

        acc = shapley.evaluate_model(h_global, test_loader)
        history.append(acc)
        times.append(time.time() - t)

        bad_n = sum(1 for c in selected if sv_dict[c] < -server.delta_q)
        extra = ""
        if rnd == 50:  extra += " ⚠ DRIFT"
        if bad_n:      extra += f" | ⛔{bad_n} bad"
        if quarantined:extra += f" | 🔒{len(quarantined)}"
        progress("FedCADS", rnd, TOTAL_ROUNDS, acc, times, extra)

    return history


# Step 9: Execute all algorithms and plot the results
print("\n" + "="*55, flush=True)
print("  TASK 3 — Real training, all 3 algorithms", flush=True)
print("  Drift injected at round 50 (clients 0-14)", flush=True)
print("="*55, flush=True)

t0          = time.time()
divfl_acc   = run_divfl()
sfedavg_acc = run_sfedavg()
fedcads_acc = run_fedcads()
print(f"\n  Done in {(time.time()-t0)/60:.1f} min", flush=True)

plt.figure(figsize=(10, 6))
plt.plot(range(1, TOTAL_ROUNDS+1), divfl_acc,   label='DivFL',                  color='blue',   alpha=0.7)
plt.plot(range(1, TOTAL_ROUNDS+1), sfedavg_acc, label='S-FedAvg',               color='orange', alpha=0.7)
plt.plot(range(1, TOTAL_ROUNDS+1), fedcads_acc, label='FedCADS-UCB (Proposed)', color='green',  linewidth=2)
plt.axvline(x=50, color='red', linestyle='--', label='Sudden Concept Drift')
plt.title('Task 3: Algorithm Robustness to Concept Drift (Simple CNN)', fontsize=14)
plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.ylim(0, 1.02)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('task3_fedcads_comparison.png', dpi=150)
plt.show()

print("\nFinal accuracy at round 100:")
print(f"  DivFL   : {divfl_acc[-1]*100:.2f}%")
print(f"  S-FedAvg: {sfedavg_acc[-1]*100:.2f}%")
print(f"  FedCADS : {fedcads_acc[-1]*100:.2f}%")